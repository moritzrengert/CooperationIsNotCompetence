#!/usr/bin/env python3
"""Run the RO all-mechanisms open-model sweep through vLLM.

GPU nodes do not need internet access. Models are served from explicit local
snapshot directories created by scripts/prepare_ro_open_models_assets.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml


@dataclass(frozen=True)
class Variant:
    slug: str
    served_model_name: str
    model_id: str
    model_path: str
    base_slug: str
    extra_body: dict[str, Any] | None
    fixed_players: dict[str, Any] | None
    model_meta: dict[str, Any]


@dataclass(frozen=True)
class ServerSpec:
    key: str
    model_id: str
    model_path: str
    served_model_name: str
    host: str
    port: int
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    vllm_args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/sweeps/ro_all_mechanisms_open_models.yaml")
    parser.add_argument("--only-models", nargs="*", default=None)
    parser.add_argument("--only-mechanisms", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--render-configs-only",
        action="store_true",
        help="Render selected configs without starting servers or making requests.",
    )
    parser.add_argument("--wait-timeout", type=int, default=1800)
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rerun experiments even when a complete existing run is found.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the whole sweep on the first failed experiment instead of logging and continuing.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a YAML mapping: {path}")
    return data


def _path_values(*values: object) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path not in paths:
            paths.append(path)
    return paths


def _has_config_json(path: Path) -> bool:
    try:
        return (path / "config.json").is_file()
    except OSError:
        return False


def local_model_path(model: dict[str, Any], defaults: dict[str, Any]) -> str:
    if model.get("local_model_dir"):
        explicit = Path(str(model["local_model_dir"])).expanduser()
        if _has_config_json(explicit):
            return str(explicit)

    candidate_roots = _path_values(
        os.environ.get("MODEL_SNAPSHOT_ROOT"),
        defaults.get("model_snapshot_root"),
        "model_snapshots",
        "model_snaps",
    )
    for root in candidate_roots:
        candidate = root / str(model["slug"])
        if _has_config_json(candidate):
            return str(candidate)

    hf_home = Path(os.environ.get("HF_HOME", "hf_cache")).expanduser()
    repo_dir = hf_home / f"models--{str(model['model_id']).replace('/', '--')}"
    ref_path = repo_dir / "refs" / "main"
    if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_dir / "snapshots" / revision
        if revision and _has_config_json(snapshot_dir):
            return str(snapshot_dir)

    root = os.environ.get("MODEL_SNAPSHOT_ROOT") or defaults.get("model_snapshot_root")
    if root:
        return str(Path(str(root)).expanduser() / str(model["slug"]))
    return str(model["model_id"])


def _model_matches_selection(model: dict[str, Any], selected: set[str]) -> bool:
    base_slug = str(model.get("slug", ""))
    if base_slug in selected:
        return True
    variants = model.get("variants")
    if not isinstance(variants, list):
        return False
    return any(str(variant.get("slug", base_slug)) in selected for variant in variants if isinstance(variant, dict))


def expand_variants(manifest: dict[str, Any]) -> list[Variant]:
    defaults = manifest.get("vllm_defaults", {})
    sweep = manifest.get("sweep") if isinstance(manifest.get("sweep"), dict) else {}
    fixed_player_ablations = sweep.get("fixed_player_ablations") if isinstance(sweep.get("fixed_player_ablations"), dict) else {}
    fixed_counts = [int(count) for count in fixed_player_ablations.get("counts", []) if int(count) > 0]
    fixed_policies = [str(policy) for policy in fixed_player_ablations.get("policies", []) if str(policy)]
    variants: list[Variant] = []
    for model in manifest["models"]:
        if model.get("enabled", True) is False:
            continue
        base_slug = str(model["slug"])
        base_meta = {key: value for key, value in model.items() if key not in {"variants", "extra_body"}}
        variant_defs = model.get("variants") or [{"slug": base_slug}]
        fixed_player_only = bool(sweep.get("fixed_player_only", False))
        for variant_def in variant_defs:
            slug = str(variant_def.get("slug", base_slug))
            if not fixed_player_only:
                variants.append(
                    Variant(
                        slug=slug,
                        served_model_name=str(model.get("served_model_name", base_slug)),
                        model_id=str(model["model_id"]),
                        model_path=local_model_path(model, defaults),
                        base_slug=base_slug,
                        extra_body=variant_def.get("extra_body", model.get("extra_body")),
                        fixed_players=None,
                        model_meta={**base_meta, **variant_def, "base_slug": base_slug},
                    )
                )
            if model.get("include_fixed_player_ablations", True) is False:
                ablations = []
            else:
                fixed_player_defs = (
                    variant_def.get("fixed_players") or model.get("fixed_players") or fixed_player_ablations.get("variants")
                )
                if isinstance(fixed_player_defs, list) and fixed_player_defs:
                    ablations = [fp for fp in fixed_player_defs if isinstance(fp, dict)]
                elif fixed_counts and fixed_policies:
                    ablations = [
                        {"count": count, "policy": policy, "placement": "randomized_player_ids"}
                        for count in fixed_counts
                        for policy in fixed_policies
                    ]
                else:
                    ablations = []
            for ablation in ablations:
                fixed_players = {
                    "count": int(ablation.get("count", 0)),
                    "policy": str(ablation.get("policy", "")),
                    "placement": str(ablation.get("placement", "randomized_player_ids")),
                }
                if fixed_players["count"] <= 0 or not fixed_players["policy"]:
                    continue
                variants.append(
                    Variant(
                        slug=slug,
                        served_model_name=str(model.get("served_model_name", base_slug)),
                        model_id=str(model["model_id"]),
                        model_path=local_model_path(model, defaults),
                        base_slug=base_slug,
                        extra_body=variant_def.get("extra_body", model.get("extra_body")),
                        fixed_players=fixed_players,
                        model_meta={**base_meta, **variant_def, "base_slug": base_slug, "fixed_players": fixed_players},
                    )
                )
    return variants


def fixed_player_suffix(variant: Variant) -> str:
    if not variant.fixed_players:
        return ""
    count = int(variant.fixed_players.get("count", 0) or 0)
    policy = _safe_slug(str(variant.fixed_players.get("policy", "none")))
    return f"fixed{count}_{policy}"

def experiment_label(variant: Variant, mechanism: str) -> str:
    suffix = fixed_player_suffix(variant)
    label = f"{variant.slug}/{mechanism}"
    return f"{label}/{suffix}" if suffix else label


def _safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "none"


def is_external_api(variant: Variant) -> bool:
    return str(variant.model_meta.get("provider", "")).lower() in {
        "openai_api", "openrouter_api", "external_openai", "external_api"
    }



def server_spec_for(manifest: dict[str, Any], variant: Variant) -> ServerSpec:
    defaults = manifest.get("vllm_defaults", {})
    model_meta = variant.model_meta
    host = str(model_meta.get("host", defaults.get("host", "127.0.0.1")))
    port = int(model_meta.get("port", defaults.get("port", 8000)))
    dtype = str(model_meta.get("dtype", defaults.get("dtype", "bfloat16")))
    max_model_len = int(model_meta.get("max_model_len", defaults.get("max_model_len", 8192)))
    gpu_memory_utilization = float(model_meta.get("gpu_memory_utilization", defaults.get("gpu_memory_utilization", 0.90)))
    vllm_args = tuple(model_meta.get("vllm_args", defaults.get("vllm_args", [])) or [])
    key = json.dumps(
        {
            "model_path": variant.model_path,
            "served_model_name": variant.served_model_name,
            "host": host,
            "port": port,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "vllm_args": list(vllm_args),
        },
        sort_keys=True,
    )
    return ServerSpec(
        key=key,
        model_id=variant.model_id,
        model_path=variant.model_path,
        served_model_name=variant.served_model_name,
        host=host,
        port=port,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        vllm_args=vllm_args,
    )


def output_dir_for(manifest: dict[str, Any], variant: Variant, mechanism: str) -> Path:
    suffix = fixed_player_suffix(variant)
    output_root = Path(manifest["sweep"]["output_root"])
    if suffix:
        return output_root / variant.slug / mechanism / suffix
    return output_root / variant.slug / mechanism


def output_path_for(manifest: dict[str, Any], variant: Variant, mechanism: str) -> Path:
    return output_dir_for(manifest, variant, mechanism) / "results.jsonl"


def expected_result_rows(manifest: dict[str, Any], variant: Variant) -> int:
    sweep = manifest["sweep"]
    meta = variant.model_meta
    repetitions = int(meta.get("repetitions", sweep.get("repetitions", 1)))
    treatments = meta.get("treatments", sweep.get("treatments", ["RESTART"]))
    chains = int(meta.get("chains_per_treatment", sweep.get("chains_per_treatment", 4)))
    generations = int(meta.get("generations_per_chain", sweep.get("generations_per_chain", 20)))
    players = int(meta.get("players_per_generation", sweep.get("players_per_generation", 5)))
    return repetitions * len(treatments) * chains * generations * players


def count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def _candidate_result_paths(result_path: Path) -> list[Path]:
    """Return possible completed run files for a configured output path.

    `src.runner.run_output_path()` materializes `outputs/foo/bar/results.jsonl`
    as `outputs/foo/bar/results_<timestamp>/results.jsonl`, so we need to look
    through the timestamped run directories under the configured parent path.
    """
    candidates: list[Path] = []
    if result_path.is_file():
        candidates.append(result_path)

    run_root = result_path.parent
    if run_root.is_dir():
        candidates.extend(sorted(run_root.glob(f"{result_path.stem}_*/results.jsonl")))
    return candidates


def is_completed(manifest: dict[str, Any], variant: Variant, mechanism: str) -> bool:
    result_path = output_path_for(manifest, variant, mechanism)
    expected = expected_result_rows(manifest, variant)
    for candidate in _candidate_result_paths(result_path):
        try:
            if count_jsonl_rows(candidate) >= expected:
                return True
        except OSError:
            continue
    return False


def render_config(manifest: dict[str, Any], variant: Variant, mechanism: str, server: ServerSpec | None) -> Path:
    sweep = manifest["sweep"]
    model_meta = variant.model_meta
    config_dir = Path(sweep["generated_config_dir"])
    config_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(sweep["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    fixed_suffix = fixed_player_suffix(variant)

    external = is_external_api(variant)
    if external:
        base_url = str(model_meta.get("base_url", "https://api.openai.com/v1"))
        model_name = variant.model_id
    else:
        if server is None:
            raise ValueError(f"Local model {variant.slug} requires a server specification")
        base_url = f"http://{server.host}:{server.port}/v1"
        model_name = variant.served_model_name
    model_block: dict[str, Any] = {
        "provider": "openai_compatible",
        "name": model_name,
        "base_url": base_url,
        "timeout": model_meta.get("timeout", sweep.get("timeout", 180)),
        "temperature": model_meta.get("temperature", sweep.get("temperature", 0.7)),
        "seed": model_meta.get("seed", sweep.get("seed", 42)),
        "max_tokens": model_meta.get("max_tokens", sweep.get("max_tokens", 512)),
        "json_response_format": model_meta.get("json_response_format", sweep.get("json_response_format", True)),
        "response_format_mode": model_meta.get("response_format_mode", sweep.get("response_format_mode", "json_object")),
    }
    if external:
        model_block["api_key_env"] = str(model_meta.get("api_key_env", "OPENAI_API_KEY"))
        model_block["response_format_fallback"] = bool(model_meta.get("response_format_fallback", False))
        model_block["resource_unavailable_retries"] = int(model_meta.get("resource_unavailable_retries", 5))
        model_block["resource_unavailable_initial_delay"] = float(
            model_meta.get("resource_unavailable_initial_delay", 1.0)
        )
        model_block["resource_unavailable_max_delay"] = float(
            model_meta.get("resource_unavailable_max_delay", 60.0)
        )
        model_block["resource_unavailable_jitter"] = float(
            model_meta.get("resource_unavailable_jitter", 0.25)
        )
        cost_guard = dict(model_meta.get("cost_guard") or {})
        cost_guard["context"] = {
            **dict(cost_guard.get("context") or {}),
            "model_slug": variant.slug,
            "mechanism": mechanism,
        }
        model_block["cost_guard"] = cost_guard
    else:
        model_block["api_key"] = "EMPTY"
    endpoint_mode = model_meta.get("endpoint_mode", sweep.get("endpoint_mode"))
    if endpoint_mode:
        model_block["endpoint_mode"] = endpoint_mode
    thinking_budget = model_meta.get("thinking_budget", sweep.get("thinking_budget"))
    if thinking_budget is not None:
        model_block["thinking_budget"] = thinking_budget
        model_block["tokenizer_name_or_path"] = variant.model_path
    if variant.extra_body:
        model_block["extra_body"] = variant.extra_body

    config = {
        "model": model_block,
        "experiment": {
            "game": sweep["game"],
            "mechanism": mechanism,
            "prompt_style": sweep.get("prompt_style", "paper"),
            "prompt_modifier": sweep.get("prompt_modifier", "none"),
            "reason_style": sweep.get("reason_style", "medium"),
            "show_generation": sweep.get("show_generation", False),
            "repetitions": model_meta.get("repetitions", sweep.get("repetitions", 1)),
            "treatments": model_meta.get("treatments", sweep.get("treatments", ["RESTART"])),
            "chains_per_treatment": model_meta.get("chains_per_treatment", sweep.get("chains_per_treatment", 4)),
            "generations_per_chain": model_meta.get("generations_per_chain", sweep.get("generations_per_chain", 20)),
            "players_per_generation": model_meta.get("players_per_generation", sweep.get("players_per_generation", 5)),
            "parallel_players": model_meta.get("parallel_players", sweep.get("parallel_players", True)),
            "player_workers": model_meta.get("player_workers", sweep.get("player_workers", 5)),
            "chain_workers": model_meta.get("chain_workers", sweep.get("chain_workers", 1)),
            "output_path": str(output_path_for(manifest, variant, mechanism)),
        },
    }
    if variant.fixed_players:
        config["experiment"]["fixed_players"] = variant.fixed_players
    if "agent_components" in sweep:
        config["experiment"]["agent_components"] = sweep["agent_components"]
    path = config_dir / f"{variant.slug}_{mechanism}{('_' + fixed_suffix) if fixed_suffix else ''}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def wait_for_vllm(
    base_url: str,
    served_model_name: str,
    timeout: int,
    process: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(f"vLLM exited before readiness with return code {return_code}")
        try:
            request = Request(f"{base_url}/models", headers={"Authorization": "Bearer EMPTY"})
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            model_ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
            if served_model_name in model_ids:
                return
            last_error = RuntimeError(f"Server is up, but {served_model_name!r} not in /models: {sorted(model_ids)}")
        except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        time.sleep(10)
    raise TimeoutError(f"vLLM did not become ready at {base_url}: {last_error}")


def vllm_env(server: ServerSpec | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")
    served_name = server.served_model_name if server is not None else ""
    model_path = server.model_path if server is not None else ""
    if (
        "qwen3.5-397b-a17b-fp8" in served_name.lower()
        or "qwen3_5_397b_a17b_fp8" in model_path.lower()
    ):
        env.setdefault("VLLM_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
    requires_cuda_compat = any(
        marker in f"{served_name} {model_path}".lower()
        for marker in (
            "deepseek-v4-flash",
            "deepseek_v4_flash",
            "minimax-m2.5",
            "minimax_m2_5",
            "hy3-fp8",
            "hy3_fp8",
        )
    )
    if requires_cuda_compat:
        compat_include = (
            Path(__file__).resolve().parents[1] / "compat" / "glibc" / "include"
        )
        # glibc 2.41 exposes C23 sinpi/cospi declarations whose exception
        # specifications conflict with this CUDA toolkit. The wrapper suppresses
        # only that declaration block while retaining all other GNU features.
        prepend_flags = env.get("NVCC_PREPEND_FLAGS", "").strip()
        compat_flag = f"-I{compat_include}"
        env["NVCC_PREPEND_FLAGS"] = f"{prepend_flags} {compat_flag}".strip()
    if server is not None and "hy3" in server.served_model_name.lower():
        # The CUDA compiler environment on this cluster omits the development
        # headers and unversioned library links needed by FlashInfer JIT builds.
        # Reuse the matching CUDA 13 files bundled with the locked environment.
        python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
        wheel_cuda_root = (
            Path(sys.prefix) / "lib" / python_tag / "site-packages" / "nvidia" / "cu13"
        )
        wheel_cuda_include = wheel_cuda_root / "include"
        wheel_cuda_lib = wheel_cuda_root / "lib"
        if wheel_cuda_include.is_dir() and wheel_cuda_lib.is_dir():
            existing_cpath = env.get("CPATH", "").strip()
            env["CPATH"] = os.pathsep.join(
                part for part in (str(wheel_cuda_include), existing_cpath) if part
            )
            compat_lib = (
                Path(env.get("XDG_CACHE_HOME", "/tmp"))
                / "llm_pgg_cuda_compat"
                / "lib"
            )
            compat_lib.mkdir(parents=True, exist_ok=True)
            for library_name in ("libcudart", "libcublas", "libcublasLt"):
                candidates = sorted(wheel_cuda_lib.glob(f"{library_name}.so.*"))
                if not candidates:
                    continue
                target = candidates[-1].resolve()
                link = compat_lib / f"{library_name}.so"
                if link.is_symlink() and link.resolve() != target:
                    link.unlink()
                if not link.exists():
                    link.symlink_to(target)
            # Also ensure libnvrtc is available for JIT builds (link into compat)
            nvrtc_candidates: list[Path] = []
            # Prefer wheel CUDA lib location
            nvrtc_candidates.extend(sorted(wheel_cuda_lib.glob("libnvrtc.so*")))
            # Fallback to common system or shared CUDA installations
            nvrtc_candidates.extend(sorted(Path("/software/ais2t").glob("**/libnvrtc.so*")))
            nvrtc_candidates.extend(sorted(Path("/software/CUDA").glob("**/libnvrtc.so*")))
            if nvrtc_candidates:
                nvrtc_target = nvrtc_candidates[-1].resolve()
                nvrtc_link = compat_lib / "libnvrtc.so"
                try:
                    if nvrtc_link.is_symlink() and nvrtc_link.resolve() != nvrtc_target:
                        nvrtc_link.unlink()
                    if not nvrtc_link.exists():
                        nvrtc_link.symlink_to(nvrtc_target)
                except OSError:
                    # If creating the symlink fails, fall back to exporting the
                    # wheel CUDA lib directory in LD_LIBRARY_PATH below so the
                    # linker can still find libnvrtc at runtime.
                    pass
            for env_name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
                existing_path = env.get(env_name, "").strip()
                env[env_name] = os.pathsep.join(
                    part for part in (str(compat_lib), str(wheel_cuda_lib), existing_path)
                    if part
                )
        # Tencent recommends TRT-LLM all-reduce for Hy3 to avoid FlashInfer's
        # multi-node workspace sizing path, including on a single TP node.
        env.setdefault("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "trtllm")
    return env


def reserve_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def start_vllm(server: ServerSpec) -> subprocess.Popen[str]:
    config_path = Path(server.model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing usable offline model snapshot: {config_path}. "
            "Run scripts/prepare_ro_open_models_assets.sh on a CPU/login node first, "
            "or materialize from the existing HF cache with --local-files-only."
        )
    cmd = [
        "uv", "run", "--offline", "vllm", "serve", server.model_path,
        "--served-model-name", server.served_model_name,
        "--host", server.host,
        "--port", str(server.port),
        "--dtype", server.dtype,
        "--max-model-len", str(server.max_model_len),
        "--gpu-memory-utilization", str(server.gpu_memory_utilization),
        *server.vllm_args,
    ]
    print("Starting vLLM:", " ".join(cmd), flush=True)
    print(
        "vLLM env:",
        json.dumps(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
                "NVCC_PREPEND_FLAGS": vllm_env(server).get("NVCC_PREPEND_FLAGS"),
                "CPATH": vllm_env(server).get("CPATH"),
                "LIBRARY_PATH": vllm_env(server).get("LIBRARY_PATH"),
                "VLLM_USE_DEEP_GEMM": vllm_env(server).get("VLLM_USE_DEEP_GEMM"),
                "VLLM_USE_DEEP_GEMM_E8M0": vllm_env(server).get(
                    "VLLM_USE_DEEP_GEMM_E8M0"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return subprocess.Popen(cmd, text=True, start_new_session=True, env=vllm_env(server))


def launch_vllm_with_retries(
    server: ServerSpec,
    wait_timeout: int,
    attempts: int | None = None,
) -> subprocess.Popen[str]:
    if attempts is None:
        deterministic_models = (
            "deepseek-v4-flash",
            "minimax-m2.5",
            "qwen3.5-397b",
            "hy3-fp8",
        )
        attempts = 1 if any(
            marker in server.served_model_name.lower()
            for marker in deterministic_models
        ) else 3

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        process: subprocess.Popen[str] | None = None
        try:
            print(
                f"Launching vLLM attempt {attempt}/{attempts} for {server.served_model_name}",
                flush=True,
            )
            process = start_vllm(server)
            wait_for_vllm(
                f"http://{server.host}:{server.port}/v1",
                server.served_model_name,
                wait_timeout,
                process=process,
            )
            return process
        except BaseException as exc:
            last_error = exc
            print(
                f"vLLM launch attempt {attempt}/{attempts} failed for {server.served_model_name}: {exc}",
                flush=True,
            )
            stop_process(process)
            if attempt < attempts:
                time.sleep(15)
    assert last_error is not None
    raise last_error


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=60)


def run_experiment(config_path: Path, output_path: Path, label: str, variant: Variant, mechanism: str) -> float:
    cmd = [sys.executable, "-m", "src.main", "--config", str(config_path)]

    print("=" * 90, flush=True)
    print(f"Next experiment: {label}", flush=True)
    print(f"  model_slug:        {variant.slug}", flush=True)
    print(f"  served_model_name: {variant.served_model_name}", flush=True)
    print(f"  base_model:        {variant.base_slug}", flush=True)
    print(f"  mechanism:         {mechanism}", flush=True)
    print(f"  fixed_players:     {variant.fixed_players or 'none'}", flush=True)
    print(f"  config:            {config_path}", flush=True)
    print(f"  output:            {output_path}", flush=True)
    print("Running command:", " ".join(cmd), flush=True)

    start = time.monotonic()
    try:
        subprocess.run(cmd, check=True)
    finally:
        elapsed = time.monotonic() - start
        print(f"Finished experiment: {label}", flush=True)
        print(f"  elapsed: {elapsed / 60:.1f} min ({elapsed:.1f} s)", flush=True)
        print("=" * 90, flush=True)

    return elapsed


def append_failure(manifest: dict[str, Any], variant: Variant, mechanism: str, config_path: Path, exc: BaseException) -> None:
    failure_path = Path(manifest["sweep"]["output_root"]) / "sweep_failures.jsonl"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time_unix": time.time(),
        "variant": variant.slug,
        "fixed_players": variant.fixed_players,
        "output_path": str(output_path_for(manifest, variant, mechanism)),
        "base_slug": variant.base_slug,
        "mechanism": mechanism,
        "config_path": str(config_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with failure_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    suffix = fixed_player_suffix(variant)
    label = f"{variant.slug}/{mechanism}" + (f"/{suffix}" if suffix else "")
    print(f"Logged failed experiment to {failure_path}: {label}", flush=True)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    sweep = manifest["sweep"]
    mechanisms = list(args.only_mechanisms or sweep["mechanisms"])
    selected = set(args.only_models or [])

    if selected:
        manifest = dict(manifest)
        manifest["models"] = [
            model
            for model in manifest.get("models", [])
            if isinstance(model, dict) and _model_matches_selection(model, selected)
        ]
    variants = expand_variants(manifest)
    if selected:
        variants = [variant for variant in variants if variant.slug in selected or variant.base_slug in selected]
    if not variants:
        raise ValueError("No model variants selected.")

    groups: dict[str, tuple[ServerSpec | None, list[Variant]]] = {}
    for variant in variants:
        if is_external_api(variant):
            server = None
            group_key = f"external:{variant.model_id}"
        else:
            server = server_spec_for(manifest, variant)
            group_key = server.key
        if group_key not in groups:
            groups[group_key] = (server, [])
        groups[group_key][1].append(variant)

    plan_name = "sweep_plan"
    if selected:
        plan_name += "_" + "_".join(_safe_slug(model) for model in sorted(selected))
    if args.only_mechanisms:
        plan_name += "_" + "_".join(_safe_slug(mechanism) for mechanism in mechanisms)
    plan_path = Path(sweep["output_root"]) / f"{plan_name}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "manifest": str(manifest_path),
        "mechanisms": mechanisms,
        "skip_completed": not args.rerun_completed,
        "continue_on_failure": not args.fail_fast,
        "groups": [
            {
                "server": server.__dict__ if server is not None else None,
                "variants": [variant.__dict__ for variant in group_variants],
            }
            for server, group_variants in groups.values()
        ],
    }
    plan_path.write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote plan to {plan_path}", flush=True)

    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return

    if args.render_configs_only:
        for server, group_variants in groups.values():
            for variant in group_variants:
                for mechanism in mechanisms:
                    rendered = render_config(manifest, variant, mechanism, server)
                    print(f"Rendered config: {rendered}", flush=True)
        return

    failed_experiments: list[str] = []
    for server, group_variants in groups.values():
        if server is None:
            pending_external: list[tuple[Variant, str, Path]] = []
            for variant in group_variants:
                for mechanism in mechanisms:
                    config_path = render_config(manifest, variant, mechanism, None)
                    if not args.rerun_completed and is_completed(manifest, variant, mechanism):
                        print(
                            f"Skipping completed experiment: {experiment_label(variant, mechanism)}",
                            flush=True,
                        )
                        continue
                    pending_external.append((variant, mechanism, config_path))
            for idx, (variant, mechanism, config_path) in enumerate(
                pending_external, start=1
            ):
                label = experiment_label(variant, mechanism)
                print(
                    f"Progress: {idx}/{len(pending_external)} external experiments for {variant.model_id}",
                    flush=True,
                )
                try:
                    run_experiment(
                        config_path,
                        output_path_for(manifest, variant, mechanism),
                        label,
                        variant,
                        mechanism,
                    )
                except subprocess.CalledProcessError as exc:
                    append_failure(manifest, variant, mechanism, config_path, exc)
                    failed_experiments.append(label)
                    if args.fail_fast:
                        raise
            continue

        server = ServerSpec(
            key=server.key,
            model_id=server.model_id,
            model_path=server.model_path,
            served_model_name=server.served_model_name,
            host=server.host,
            port=reserve_free_port(server.host),
            dtype=server.dtype,
            max_model_len=server.max_model_len,
            gpu_memory_utilization=server.gpu_memory_utilization,
            vllm_args=server.vllm_args,
        )
        # If Slurm allocated fewer GPUs than the requested tensor-parallel-size,
        # reduce the requested tensor-parallel-size to the available GPU count
        # so vLLM doesn't fail on startup. Prefer SLURM_JOB_GPUS, then
        # CUDA_VISIBLE_DEVICES, then torch.cuda if available.
        avail_gpus = 0
        slurm_gpus = os.environ.get("SLURM_JOB_GPUS", "").strip()
        if slurm_gpus:
            avail_gpus = len([p for p in re.split(r"[,:]", slurm_gpus) if p.strip()])
        if avail_gpus == 0:
            cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if cuda_vis:
                avail_gpus = len([p for p in re.split(r"[,:]", cuda_vis) if p.strip()])
        if avail_gpus == 0:
            try:
                import torch

                avail_gpus = torch.cuda.device_count()
            except Exception:
                avail_gpus = 0

        if avail_gpus > 0:
            args_list = list(server.vllm_args)
            if "--tensor-parallel-size" in args_list:
                idx = args_list.index("--tensor-parallel-size")
                if idx + 1 < len(args_list):
                    try:
                        requested = int(args_list[idx + 1])
                        if requested > avail_gpus:
                            print(
                                f"Adjusting --tensor-parallel-size from {requested} to available GPUs {avail_gpus}",
                                flush=True,
                            )
                            args_list[idx + 1] = str(avail_gpus)
                            server = ServerSpec(
                                key=server.key,
                                model_id=server.model_id,
                                model_path=server.model_path,
                                served_model_name=server.served_model_name,
                                host=server.host,
                                port=server.port,
                                dtype=server.dtype,
                                max_model_len=server.max_model_len,
                                gpu_memory_utilization=server.gpu_memory_utilization,
                                vllm_args=tuple(args_list),
                            )
                    except Exception:
                        pass
        pending: list[tuple[Variant, str, Path]] = []
        for variant in group_variants:
            for mechanism in mechanisms:
                config_path = render_config(manifest, variant, mechanism, server)
                if not args.rerun_completed and is_completed(manifest, variant, mechanism):
                    label = experiment_label(variant, mechanism)
                    print(f"Skipping completed experiment: {label}", flush=True)
                    continue
                pending.append((variant, mechanism, config_path))
        if not pending:
            print(f"Skipping server for {server.served_model_name}: no pending experiments", flush=True)
            continue

        process: subprocess.Popen[str] | None = None
        try:
            process = launch_vllm_with_retries(server, args.wait_timeout)
            for idx, (variant, mechanism, config_path) in enumerate(pending, start=1):
                label = experiment_label(variant, mechanism)
                output_path = output_path_for(manifest, variant, mechanism)
                print(
                    f"Progress: {idx}/{len(pending)} pending experiments for server {server.served_model_name}",
                    flush=True,
                )
                experiment_attempts = (
                    1 if "qwen3.5-397b" in server.served_model_name.lower() else 2
                )
                attempt = 1
                while True:
                    try:
                        run_experiment(config_path, output_path, label, variant, mechanism)
                        break
                    except subprocess.CalledProcessError as exc:
                        if attempt >= experiment_attempts:
                            append_failure(manifest, variant, mechanism, config_path, exc)
                            failed_experiments.append(label)
                            if args.fail_fast:
                                raise
                            break
                        print(
                            f"Experiment failed for {label}; restarting vLLM and retrying once",
                            flush=True,
                        )
                        stop_process(process)
                        process = launch_vllm_with_retries(server, args.wait_timeout)
                        attempt += 1
        finally:
            stop_process(process)

    if failed_experiments:
        raise RuntimeError(f"{len(failed_experiments)} experiment(s) failed: {', '.join(failed_experiments)}")


if __name__ == "__main__":
    main()
