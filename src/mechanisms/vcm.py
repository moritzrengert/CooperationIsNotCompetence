from src.games.ro_public_good import (
    RO_MAX_CONTRIBUTION,
    RO_MIN_CONTRIBUTION,
    contribution_share,
    paper_payoff_block,
    paper_period_label,
    public_good_payoff,
)
from src.mechanisms.base import Mechanism
from src.mechanisms.ro_display import previous_period_for_prompt, rows_in_player_view


def _format_history(history: list[list[dict]], player: int | None) -> str:
    """Render exactly the previous-period information shown by the VCM screen."""
    previous = previous_period_for_prompt(history)
    if previous is None:
        return "In the first period this block is empty."

    previous_period, rows = previous
    ordered = rows_in_player_view(rows, player)
    contributions = [row["implemented_contribution"] for row in ordered]
    own_previous = contributions[0]
    return (
        f"Previous-period block (period {previous_period}): contributions for Player 1 (Yourself), "
        f"Player 2, Player 3, Player 4, and Player 5 = {contributions}.\n"
        "Defaults on the current screen: your contribution field is prefilled with your previous contribution "
        f"({own_previous}), and the calculator is prefilled with the previous contribution profile "
        f"{contributions}. You may keep or change any prefilled value."
    )


class VCMMechanism(Mechanism):
    """Voluntary Contribution Mechanism from Oechssler et al. (2022)."""

    name = "vcm"

    def augment_observation(self, observation, context):
        observation = observation.copy()
        observation["choice_options"] = list(range(RO_MIN_CONTRIBUTION, RO_MAX_CONTRIBUTION + 1))
        return observation

    def augment_prompt(self, prompt, observation):
        # The shared game block comes from Appendix D through ROPublicGoodGame.
        # This file adds Appendix D's VCM-specific payoff/screen material plus
        # the period, previous-screen state, and machine-readable response block.
        history_text = _format_history(observation.get("history", []), observation.get("player"))
        period_label = paper_period_label(observation.get("period_number", 1))
        payoff_block = paper_payoff_block(self.name)
        reason_instruction = observation["reason_instruction"]
        reason_field = ""
        if reason_instruction != "optional empty string":
            reason_field = f',\n            "reason": "{reason_instruction}"'

        return f"""{prompt}

            {payoff_block}

            Structure of the Screen

            You have a printed example for the structure of the screen of the program you will make your decisions with in
            each period. The screen is divided into three blocks.

            The upper left block contains a calculator. Here you can test actions for you and the four other players (by
            default the computer will show here the decisions of the last period. Of course you can change them). Once you
            selected an action for every player you can press the button “Calculate payoff!” and the computer will calculate
            the payoff you would obtain in this case.

            In the upper right block you enter the action that will be relevant for your payoff (here the computer will show
            as a default your contribution of the last period as well. If you want to change it, you have to do so now). Below
            there is a red button. When you press this button you submit your decision and leave the screen. Only when all
            players have pressed the red button the experiment continues. A clock on the upper right gives you a hint until
            when you should have made a decision. If the time runs out this has no consequences.

            In the bottom block you will see from the second period onwards the actions of all players in the previous period.
            In the first period this block will be empty.

            {period_label}

            Information displayed on the current screen:

            {history_text}

            Return ONLY valid JSON with this schema:
            {{
            "choice": integer from 0 to 10{reason_field}
            }}
        """

    def resolve(self, decisions, context):
        contributions = [int(d.choice) for d in decisions]
        total = sum(contributions)
        return {
            "rows": [
                {
                    "implemented_contribution": contribution,
                    "contribution_share": contribution_share(contribution),
                    "contributes": contribution > 0,
                    "total_contribution": total,
                    "total_contributors": sum(value > 0 for value in contributions),
                    "payoff": public_good_payoff(contribution, total),
                }
                for contribution in contributions
            ],
            "contributors": sum(value > 0 for value in contributions),
            "total_contribution": total,
        }
