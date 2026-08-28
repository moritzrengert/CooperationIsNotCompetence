import json
import hashlib
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from src.agents import Decision, LLMAgent
from src.agent_components import AgentComponents
from src.fixed_players import decision_for_fixed_policy, parse_fixed_player_config
from src.game import Treatment
from src.games import get_game
from src.mechanisms.ccf import compile_ccf_source
from src.mechanisms import get_mechanism
from src.prompts import REASON_INSTRUCTIONS, apply_prompt_modifier, reason_instruction


def run_output_path(output_path: str) -> Path:
    path = Path(output_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{path.stem}_{timestamp}" if path.suffix else f"{path.name}_{timestamp}"
    return path.parent / run_name / "results.jsonl"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None



class ExperimentRunner:
    def __init__(
        self, llm, treatments: Optional[list[str]] = None, chains_per_treatment: int = 1,
        generations_per_chain: int = 20, players_per_generation: int = 5,
        initial_reserve: float = 183, output_path: str = "outputs/results.jsonl",
        prompt_style: str = "human_table", repetitions: int = 1,
        parallel_players: bool = False, player_workers: Optional[int] = None,
        chain_workers: int = 1,
        max_attempts: int = 3, prompt_modifier: Optional[str] = None,
        config_path: Optional[str] = None, show_generation: bool = False,
        mechanism: str = "fischer_2004", game: str = "fischer_2004",
        reason_style: str = "short", fixed_players: Optional[dict] = None,
        agent_components: Optional[dict] = None,
    ):
        self.llm = llm
        self.game_name = game
        self.mechanism_name = mechanism
        self.game = get_game(game)
        self.mechanism = get_mechanism(mechanism)
        self.treatments = [Treatment(t) for t in (treatments or ["RESTART"])]
        self.chains = chains_per_treatment
        self.generations = generations_per_chain
        self.players = players_per_generation
        self.initial_reserve = initial_reserve
        self.output_path = run_output_path(output_path)
        self.run_dir = self.output_path.parent
        self.config_path = Path(config_path) if config_path is not None else None
        self.prompt_style = prompt_style
        self.repetitions = repetitions
        self.parallel_players = parallel_players
        self.player_workers = player_workers or players_per_generation
        self.chain_workers = int(chain_workers)
        if self.chain_workers < 1:
            raise ValueError("chain_workers must be >= 1.")
        self.max_attempts = max_attempts
        self.prompt_modifier = prompt_modifier
        self.started_at_utc = datetime.now(timezone.utc)
        self.fixed_players = parse_fixed_player_config(
            fixed_players,
            total_players=self.players,
            game_name=self.game_name,
            mechanism_name=self.mechanism_name,
        )
        if not isinstance(show_generation, bool):
            raise ValueError("show_generation must be a boolean.")
        self.show_generation = show_generation
        if reason_style not in REASON_INSTRUCTIONS:
            raise ValueError(f"Unknown reason_style: {reason_style}. Available: {sorted(REASON_INSTRUCTIONS)}")
        self.reason_style = reason_style
        self.agent_components = AgentComponents.from_config(agent_components)

    @property
    def meta(self):
        return {
            "game": self.game_name,
            "mechanism": self.mechanism_name,
            "prompt_style": self.prompt_style,
            "prompt_modifier": self.prompt_modifier,
            "reason_style": self.reason_style,
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "seed": self.llm.seed,
            "show_generation": self.show_generation,
            **self.agent_components.metadata(),
        }

    @property
    def fixed_meta(self):
        return {
            "fixed_count": self.fixed_players.count,
            "fixed_policy": self.fixed_players.row_policy,
            "fixed_placement": self.fixed_players.placement,
        }

    @property
    def expected_rows(self) -> int:
        return self.repetitions * len(self.treatments) * self.chains * self.generations * self.players

    def _write_run_metadata(self) -> None:
        metadata = {
            "results_path": str(self.output_path),
            "run_dir": str(self.run_dir),
            "started_at_utc": self.started_at_utc.isoformat(),
            "expected_rows": self.expected_rows,
            "meta": self.meta,
            "experiment": {
                "game": self.game_name,
                "mechanism": self.mechanism_name,
                "treatments": [t.value for t in self.treatments],
                "chains_per_treatment": self.chains,
                "generations_per_chain": self.generations,
                "players_per_generation": self.players,
                "initial_reserve": self.initial_reserve,
                "prompt_style": self.prompt_style,
                "repetitions": self.repetitions,
                "parallel_players": self.parallel_players,
                "player_workers": self.player_workers,
                "chain_workers": self.chain_workers,
                "max_attempts": self.max_attempts,
                "prompt_modifier": self.prompt_modifier,
                "show_generation": self.show_generation,
                "reason_style": self.reason_style,
                "fixed_players": self.fixed_meta,
                "agent_components": self.agent_components.configuration(),
            },
            "environment": {
                "python": platform.python_version(),
                "git_commit": _git_commit(),
            },
        }
        if self.config_path is not None:
            copied = self.run_dir / self.config_path.name
            metadata["config"] = {
                "source_path": str(self.config_path),
                "copied_path": str(copied),
                "sha256": _sha256(copied) if copied.is_file() else None,
            }
        (self.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def effective_seed(self, rep: int, chain: int, gen: int, player: int) -> Optional[int]:
        if self.llm.seed is None:
            return None
        return self.llm.seed + 100000 * rep + 1000 * chain + 10 * gen + player

    def fixed_player_ids_for(self, rep: int, chain: int) -> set[int]:
        return self.fixed_players.player_ids(repetition=rep, chain=chain, seed=self.llm.seed)

    @staticmethod
    def prompt_observation(observation: dict) -> dict:
        show_generation = observation.get("show_generation", False)
        prompt_observation = {
            key: value for key, value in observation.items()
            if key not in {"effective_seed", "show_generation"}
        }
        if not show_generation:
            prompt_observation.pop("generation", None)
            prompt_observation.pop("total_generations", None)
        return prompt_observation

    def build_context(self, treatment, rep, chain, gen, reserve, history):
        fixed_player_ids = self.fixed_player_ids_for(rep, chain)
        return {
            "meta": self.meta,
            "game_name": self.game_name,
            "mechanism_name": self.mechanism_name,
            "treatment": treatment,
            "repetition": rep,
            "chain": chain,
            "generation": gen,
            "players": self.players,
            "reserve": reserve,
            "history": history,
            "effective_seed": self.effective_seed,
            "fixed_player_ids": fixed_player_ids,
            **self.fixed_meta,
        }

    def build_observation(self, context):
        observation = self.game.observation(context)
        if self.show_generation:
            observation["total_generations"] = self.generations
        return self.mechanism.augment_observation(observation, context)

    def build_prompt(self, observation, state=None):
        prompt_observation = self.prompt_observation(observation)
        instruction = reason_instruction(self.reason_style)
        prompt = self.game.prompt(prompt_observation, instruction)
        prompt = self.mechanism.augment_prompt(
            prompt,
            {
                **prompt_observation,
                "reason_instruction": instruction,
                "prompt_modifier": self.prompt_modifier or "none",
            },
        )
        prompt = apply_prompt_modifier(prompt, self.prompt_modifier or "none")
        state = self.agent_components.initial_state() if state is None else state
        return self.agent_components.augment_prompt(prompt, state)

    def decide(self, agent, prompt, observation):
        err = None
        for attempt in range(self.max_attempts):
            retry_observation = observation
            if attempt and observation.get("effective_seed") is not None:
                retry_observation = {
                    **observation,
                    "effective_seed": observation["effective_seed"] + 1_000_000 * attempt,
                }
            try:
                decision = agent.decide(prompt, retry_observation)
                self._validate_decision(decision, retry_observation)
                decision.retry_count = attempt
                return decision
            except ValueError as exc:
                err = exc
        if observation.get("ccf_source_required"):
            return self._fallback_ccf_decision(agent.agent_id, err)
        raise RuntimeError(f"{agent.agent_id} failed after {self.max_attempts} attempts") from err

    @staticmethod
    def _validate_decision(decision: Decision, observation: dict) -> None:
        if observation.get("ccf_source_required"):
            compile_ccf_source(decision.ccf_source or "")

    def _fallback_ccf_decision(self, agent_id: str, err: Exception | None) -> Decision:
        source = "def ccf(others_total):\n    return 0"
        error_text = str(err) if err is not None else "unknown CCF validation failure"
        payload = {
            "ccf_source": source,
            "reason": "Fallback after repeated invalid CCF submissions.",
            "fallback_used": True,
            "fallback_error": error_text,
        }
        return Decision(
            choice=0,
            prediction=None,
            raw_response=json.dumps(payload),
            reason="Fallback after repeated invalid CCF submissions.",
            ccf_source=source,
            fallback_used=True,
            fallback_error=f"{agent_id}: {error_text}",
            parser_mode="fallback_ccf",
            retry_count=max(self.max_attempts - 1, 0),
        )

    def fixed_decision(self, observation):
        return decision_for_fixed_policy(
            policy=self.fixed_players.policy,
            mechanism_name=self.mechanism_name,
            generation=observation["generation"],
            player=observation["player"],
            repetition=observation["repetition"],
            chain=observation["chain_id"],
            seed=observation.get("effective_seed"),
        )

    def decide_all(self, agents, observation, states=None):
        rep = observation["repetition"]
        chain = observation["chain_id"]
        gen = observation["generation"]
        fixed_player_ids = self.fixed_player_ids_for(rep, chain)

        if states is None:
            states = {player: self.agent_components.initial_state() for player in agents}

        decisions = [None] * self.players
        items = []
        for player in range(1, self.players + 1):
            obs = observation.copy()
            obs["player"] = player
            obs["effective_seed"] = self.effective_seed(rep, chain, gen, player)
            if player in fixed_player_ids:
                decision = self.fixed_decision(obs)
                self.agent_components.mark_inactive(decision)
                decisions[player - 1] = decision
                continue
            agent = agents[player]
            state = states[player]
            prompt = self.build_prompt(obs, state)
            items.append((player, agent, state, prompt, obs))

        if not self.parallel_players:
            for player, agent, state, prompt, obs in items:
                decision = self.decide(agent, prompt, obs)
                self.agent_components.apply(state, decision)
                decisions[player - 1] = decision
            return decisions

        if items:
            with ThreadPoolExecutor(max_workers=min(self.player_workers, len(items))) as ex:
                llm_decisions = list(ex.map(lambda x: self.decide(x[1], x[3], x[4]), items))
            for (player, _, state, _, _), decision in zip(items, llm_decisions):
                self.agent_components.apply(state, decision)
                decisions[player - 1] = decision
        return decisions

    def _run_chain(self, rep, treatment, chain, on_generation: Optional[Callable[[], None]] = None):
        reserve = self.initial_reserve
        history = []
        chain_rows = []
        fixed_player_ids = self.fixed_player_ids_for(rep, chain)
        agents = {
            p: LLMAgent(f"{treatment}-r{rep}-c{chain}-p{p}", self.llm, self.agent_components)
            for p in range(1, self.players + 1)
            if p not in fixed_player_ids
        }
        states = {p: self.agent_components.initial_state() for p in agents}

        for gen in range(1, self.generations + 1):
            context = self.build_context(treatment, rep, chain, gen, reserve, history)
            observation = self.build_observation(context)
            decisions = self.decide_all(agents, observation, states)
            outcome = self.mechanism.resolve(decisions, context)
            rows, next_state = self.game.rows(context, decisions, outcome)
            chain_rows.extend(rows)

            if "reserve" in next_state:
                reserve = next_state["reserve"]
            if "history_entry" in next_state:
                history.append(next_state["history_entry"])
            if on_generation is not None:
                on_generation()

        return chain_rows

    def _chain_tasks(self):
        return [
            (rep, treatment, chain)
            for rep in range(1, self.repetitions + 1)
            for treatment in self.treatments
            for chain in range(1, self.chains + 1)
        ]

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path is not None:
            shutil.copy2(self.config_path, self.run_dir / self.config_path.name)
        self._write_run_metadata()

        total_steps = self.repetitions * len(self.treatments) * self.chains * self.generations
        with self.output_path.open("w", encoding="utf-8") as f:
            with tqdm(total=total_steps, desc="Experiments", leave=True) as pbar:
                tasks = self._chain_tasks()
                update_progress = lambda: pbar.update(1)
                if self.chain_workers == 1:
                    for task in tasks:
                        for row in self._run_chain(*task, on_generation=update_progress):
                            f.write(json.dumps(row) + "\n")
                else:
                    workers = min(self.chain_workers, len(tasks))
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        run_chain = lambda task: self._run_chain(*task, on_generation=update_progress)
                        for chain_rows in ex.map(run_chain, tasks):
                            for row in chain_rows:
                                f.write(json.dumps(row) + "\n")

        print(f"Done. Results saved to {self.output_path}")
