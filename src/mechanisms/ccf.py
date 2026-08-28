from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from types import SimpleNamespace

from src.games.ro_public_good import contribution_share
from src.mechanisms.base import Mechanism
from src.mechanisms.ccm import scaled_public_good_payoff

CCF_MIN_OTHERS_TOTAL = 0
CCF_MAX_OTHERS_TOTAL = 40
CCF_MAX_CONTRIBUTION = 10

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sum": sum,
}
SAFE_MATH = SimpleNamespace(
    ceil=math.ceil,
    floor=math.floor,
    sqrt=math.sqrt,
    log=math.log,
    exp=math.exp,
)


class CCFValidationError(ValueError):
    pass


class _CCFSafetyValidator(ast.NodeVisitor):
    _ALLOWED_NODES = {
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.AnnAssign,
        ast.Expr,
        ast.If,
        ast.IfExp,
        ast.Compare,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Subscript,
        ast.Call,
        ast.Attribute,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    }
    _ALLOWED_CALLS = set(SAFE_BUILTINS)
    _ALLOWED_MATH_ATTRS = {"ceil", "floor", "sqrt", "log", "exp"}

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in self._ALLOWED_NODES:
            raise CCFValidationError(f"Unsupported syntax in CCF source: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        if len(node.body) != 1 or not isinstance(node.body[0], ast.FunctionDef):
            raise CCFValidationError("CCF source must contain exactly one top-level function definition.")
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name != "ccf":
            raise CCFValidationError("CCF function must be named 'ccf'.")
        if len(node.args.args) != 1 or node.args.args[0].arg != "others_total":
            raise CCFValidationError("CCF function must take exactly one argument named 'others_total'.")
        if node.decorator_list:
            raise CCFValidationError("Decorators are not allowed in CCF source.")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise CCFValidationError("Double-underscore names are not allowed in CCF source.")
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise CCFValidationError("Private attributes are not allowed in CCF source.")
        if not isinstance(node.value, ast.Name) or node.value.id != "math" or node.attr not in self._ALLOWED_MATH_ATTRS:
            raise CCFValidationError("Only selected math functions are available in CCF source.")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = node.func
        if isinstance(target, ast.Name):
            if target.id not in self._ALLOWED_CALLS:
                raise CCFValidationError(f"Call to unsupported function {target.id!r} in CCF source.")
        elif isinstance(target, ast.Attribute):
            self.visit_Attribute(target)
        else:
            raise CCFValidationError("Unsupported call target in CCF source.")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)


@dataclass(frozen=True)
class CompiledCCF:
    source: str
    outputs: tuple[int, ...]

    def contribution(self, others_total: int) -> int:
        return self.outputs[others_total]


def _coerce_output(value: object, others_total: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise CCFValidationError(
        f"CCF must return an integer contribution for others_total={others_total}; got {value!r}."
    )


def _normalize_source(source: str) -> str:
    normalized = str(source).strip()
    if not normalized:
        return normalized
    if "\\n" in normalized and "\n" not in normalized:
        normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n")
    if "\\t" in normalized and "\t" not in normalized:
        normalized = normalized.replace("\\t", "\t")
    return normalized


def compile_ccf_source(source: str) -> CompiledCCF:
    normalized_source = _normalize_source(source)
    try:
        tree = ast.parse(normalized_source, mode="exec")
    except SyntaxError as exc:
        raise CCFValidationError(f"Invalid Python syntax in CCF source: {exc.msg}") from exc
    _CCFSafetyValidator().visit(tree)

    namespace: dict[str, object] = {"__builtins__": SAFE_BUILTINS, "math": SAFE_MATH}
    exec(compile(tree, "<ccf>", "exec"), namespace, namespace)
    func = namespace.get("ccf")
    if not callable(func):
        raise CCFValidationError("CCF source did not define a callable 'ccf' function.")

    outputs: list[int] = []
    previous = None
    for others_total in range(CCF_MIN_OTHERS_TOTAL, CCF_MAX_OTHERS_TOTAL + 1):
        try:
            raw_value = func(others_total)
        except Exception as exc:
            raise CCFValidationError(
                f"CCF execution failed for others_total={others_total}: {exc}"
            ) from exc
        value = _coerce_output(raw_value, others_total)
        if not 0 <= value <= CCF_MAX_CONTRIBUTION:
            raise CCFValidationError(
                f"CCF contribution must stay in 0..{CCF_MAX_CONTRIBUTION}; got {value} for others_total={others_total}."
            )
        if previous is not None and value < previous:
            raise CCFValidationError(
                f"CCF must be weakly increasing in others_total; got {previous} then {value}."
            )
        outputs.append(value)
        previous = value
    return CompiledCCF(source=normalized_source, outputs=tuple(outputs))


def ccf_resolve(compiled_ccfs: list[CompiledCCF]) -> tuple[list[int], int]:
    contributions = [CCF_MAX_CONTRIBUTION] * len(compiled_ccfs)
    while True:
        updated = [
            compiled.contribution(sum(contributions) - contributions[idx])
            for idx, compiled in enumerate(compiled_ccfs)
        ]
        if updated == contributions:
            total = sum(updated)
            return updated, total
        contributions = updated


def ccf_metadata(compiled: CompiledCCF) -> dict[str, object]:
    outputs = list(compiled.outputs)
    return {
        "ccf_source": compiled.source,
        "ccf_outputs": outputs,
        "ccf_min_contribution": min(outputs),
        "ccf_max_contribution": max(outputs),
        "ccf_is_conditional": len(set(outputs)) > 1,
    }


class CCFMechanism(Mechanism):
    name = "ccf"

    def augment_observation(self, observation, context):
        observation = observation.copy()
        observation["ccf_others_total_options"] = list(range(CCF_MIN_OTHERS_TOTAL, CCF_MAX_OTHERS_TOTAL + 1))
        observation["ccf_contribution_options"] = list(range(CCF_MAX_CONTRIBUTION + 1))
        observation["ccf_source_required"] = True
        return observation

    def augment_prompt(self, prompt, observation):
        history = observation.get("history", [])
        player = observation.get("player")
        history_text = "In the first period this history is empty."
        if history:
            lines = []
            for period, rows in enumerate(history, 1):
                contributions = [row["implemented_contribution"] for row in rows]
                total = sum(contributions)
                your_payoff = next((row.get("payoff") for row in rows if row.get("player") == player), None)
                lines.append(
                    f"P{period}: implemented={contributions}; total={total}; your_payoff={your_payoff}"
                )
            history_text = "Previous periods in your group:\n" + "\n".join(lines)

        reason_instruction = observation["reason_instruction"]
        return f"""{prompt}

            The Conditional Commitment Function (CCF) mechanism is active.
            You must submit a single Python function named ccf with interface:
                def ccf(others_total: int) -> int:
            Here others_total is the total contribution of the other four players, so it ranges from 0 to 40.
            Your function must return your own contribution in integer points from 0 to 10.
            The function must be weakly increasing: if others_total increases, your returned contribution may stay the same or increase, but never decrease.

            The submitted function is binding. The computer computes the largest mutually feasible contribution profile implied by all five submitted functions.
            Your payoff is then: 10 - your implemented contribution + 0.4 * total implemented contribution.

            Sandbox rules:
            - No imports are allowed.
            - A restricted math object is already available with ceil, floor, sqrt, log, and exp.
            - No file, network, or environment access exists.
            - Keep the function pure and deterministic.

            Practical examples:
            - Always keep everything: `def ccf(others_total): return 0`
            - Always contribute everything: `def ccf(others_total): return 10`
            - Contribute 10 only if others contribute at least 30: `def ccf(others_total): return 10 if others_total >= 30 else 0`

            From period two on the implemented contributions of all players of all previous periods and your payoff in those periods will be displayed. {history_text}

            Return ONLY valid JSON with this schema:
            {{
            "ccf_source": "string containing the complete Python function definition",
            "reason": "{reason_instruction}"
            }}
            """

    def resolve(self, decisions, context):
        compiled = [compile_ccf_source(decision.ccf_source or "") for decision in decisions]
        contributions, total = ccf_resolve(compiled)
        contributors = sum(contribution > 0 for contribution in contributions)
        return {
            "rows": [
                {
                    **ccf_metadata(compiled_ccf),
                    "implemented_contribution": contribution,
                    "contribution_share": contribution_share(contribution),
                    "contributes": contribution > 0,
                    "total_contribution": total,
                    "total_contributors": contributors,
                    "payoff": scaled_public_good_payoff(contribution, total),
                }
                for compiled_ccf, contribution in zip(compiled, contributions)
            ],
            "contributors": contributors,
            "total_contribution": total,
        }
