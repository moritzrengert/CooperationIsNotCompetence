from src.mechanisms.activation import resolve_activation
from src.mechanisms.base import Mechanism


RO_GAME_NAMES = {"ro_public_good", "ro_bccm", "ro_ccm", "ro_vcm"}


def is_ro_public_good_game(game_name: str | None) -> bool:
    return game_name in RO_GAME_NAMES


def bccm_resolve(offers: list[int], conditions: list[int], fallback: int = 1) -> tuple[list[int], int]:
    active, active_count = resolve_activation(conditions)
    return [offer if is_active else fallback for offer, is_active in zip(offers, active)], active_count


def conditional_contribution_resolve(conditions: list[int]) -> tuple[list[bool], int]:
    # Paper condition c means: contribute if at least c other players contribute.
    # Equivalently, the required total number of contributors is c + 1.
    return resolve_activation([condition + 1 for condition in conditions])


def public_good_payoff(contributes: bool, total_contributors: int) -> int:
    return (0 if contributes else 10) + 6 * total_contributors


class BCCMMechanism(Mechanism):
    """Binary conditional contribution mechanism adapted to the active game.

    The same mechanism idea is used in both supported games, but the action
    interface is game-dependent:

    - Fischer CPR adaptation: agents choose an effort offer plus a condition;
      inactive offers are replaced by fallback effort 1.
    - RO public-good reproduction: agents choose only the paper condition;
      the mechanism resolves whether each player contributes.
    """

    name = "bccm"

    def augment_observation(self, observation, context):
        observation = observation.copy()
        game_name = context.get("game_name")
        if is_ro_public_good_game(game_name):
            observation["condition_options"] = list(range(context["players"] + 1))
        else:
            observation["condition_options"] = list(range(1, context["players"] + 2))
        return observation

    def augment_prompt(self, prompt, observation):
        if is_ro_public_good_game(observation.get("game")):
            return self._ro_prompt(prompt, observation)
        return self._fischer_prompt(prompt, observation)

    def resolve(self, decisions, context):
        if is_ro_public_good_game(context.get("game_name")):
            contributes, total = conditional_contribution_resolve([d.condition for d in decisions])
            return {
                "rows": [
                    {
                        "condition": decision.condition,
                        "contributes": contributes_now,
                        "total_contributors": total,
                        "payoff": public_good_payoff(contributes_now, total),
                    }
                    for decision, contributes_now in zip(decisions, contributes)
                ],
                "contributors": total,
            }

        implemented_choices, contributors = bccm_resolve(
            [d.choice for d in decisions],
            [d.condition for d in decisions],
        )
        return {
            "implemented_choices": implemented_choices,
            "contributors": contributors,
        }

    @staticmethod
    def _fischer_prompt(prompt, observation):
        reason_instruction = observation["reason_instruction"]
        return f"""
            In addition to your effort choice, every player chooses one of four implementation conditions:

            1 = Your effort choice is implemented in any case.
            2 = Your effort choice is implemented only if at least one other player also has his effort choice implemented.
            3 = Your effort choice is implemented only if both other players also have their effort choices implemented.
            4 = Your effort choice is implemented in no case.

            The computer selects the highest possible number of players whose effort choices can be implemented
            without violating the condition of any selected player. Selected players receive their chosen effort.
            Players who are not selected receive effort 1. The payoff table below is evaluated using the final
            implemented efforts, not merely the efforts initially offered.

            Example 1: 2 players choose condition "2" and the other player chooses condition "4". Then those 2
            players who chose condition "2" have their effort choices implemented. The condition-4 player receives effort 1.

            Example 2: 2 players choose condition "3" and the other player chooses condition "4". Then no player
            has his effort choice implemented, so all three players receive effort 1.

            {prompt}

            Because the conditional-effort mechanism is active, return ONLY valid JSON with this schema:
            {{
            "choice": integer from 1 to 8,
            "condition": integer from 1 to 4,
            "prediction": integer from 2 to 16,
            "reason": "{reason_instruction}"
            }}
            """

    @staticmethod
    def _ro_prompt(prompt, observation):
        history = observation.get("history", [])
        player = observation.get("player")
        history_text = "In the first period this history is empty."
        if history:
            lines = []
            for period, rows in enumerate(history, 1):
                conditions = [row["condition"] for row in rows]
                your_payoff = next((row.get("payoff") for row in rows if row.get("player") == player), None)
                lines.append(f"P{period}: conditions={conditions}; your_payoff={your_payoff}")
            history_text = "Previous periods in your group:\n" + "\n".join(lines)
        reason_instruction = observation["reason_instruction"]
        return f"""{prompt}

            Every player can choose in every period between six different conditions:
            0 = Contribute in any case.
            1 = Contribute only if at least one other player contributes, too.
            2 = Contribute only if at least two other players contribute, too.
            3 = Contribute only if at least three other players contribute, too.
            4 = Contribute only if all four other players contribute, too.
            5 = Contribute in no case.

            The computer selects the highest amount of players, which can contribute to the project, without
            violation of the condition of any player. These players will then automatically contribute to the
            project. The other players will not contribute.

            Example 1: 3 players choose condition "1" and the other two players choose condition "5". Then those
            3 players, who chose condition "1" will contribute to the project.

            Example 2: 3 players choose condition "3" and the other two players choose condition "5". Then no
            player will contribute to the project.

            From period two on the conditions of all players of all previous periods and your payoff in those
            periods will be displayed. {history_text}

            Return ONLY valid JSON with this schema:
            {{
            "condition": integer from 0 to 5,
            "reason": "{reason_instruction}"
            }}
            """


# Backward-compatible alias for older imports/config mapping.
ROBCCMMechanism = BCCMMechanism
