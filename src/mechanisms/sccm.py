from src.games.ro_public_good import contribution_share, paper_payoff_block, paper_period_label, public_good_payoff
from src.mechanisms.base import Mechanism
from src.mechanisms.ccm import ccm_resolve
from src.mechanisms.ro_display import previous_period_for_prompt, rows_in_player_view


def _format_history(history: list[list[dict]], player: int | None) -> str:
    """Render exactly the previous-period information shown by the SCCM screen."""
    previous = previous_period_for_prompt(history)
    if previous is None:
        return "In the first period this block is empty."

    previous_period, rows = previous
    ordered = rows_in_player_view(rows, player)
    messages = [(row.get("contribution_1"), row.get("threshold_1")) for row in ordered]
    implemented = [row["implemented_contribution"] for row in ordered]
    own_previous = messages[0]
    return (
        f"Previous-period block (period {previous_period}): conditions (X, Y) for Player 1 (Yourself), "
        f"Player 2, Player 3, Player 4, and Player 5 = {messages}; implemented contributions in the same "
        f"order = {implemented}.\n"
        "Defaults on the current screen: your condition field is prefilled with your own previous condition "
        f"{own_previous}, and the calculator is prefilled with the five previous conditions {messages}. "
        "You may keep or change any prefilled value."
    )


class SCCMMechanism(Mechanism):
    """Simple Conditional Contribution Mechanism from Oechssler et al. (2022)."""

    name = "sccm"

    def augment_observation(self, observation, context):
        observation = observation.copy()
        observation["contribution_options"] = list(range(11))
        observation["threshold_options"] = list(range(10 * context["players"] + 1))
        observation["offer_count"] = 1
        return observation

    def augment_prompt(self, prompt, observation):
        # The shared game block comes from the wording common to Appendices B/F.
        # This file adds Appendix F's SCCM-specific rules, payoff/screen material,
        # and the period, previous-screen state, and machine-readable response block.
        history_text = _format_history(observation.get("history", []), observation.get("player"))
        period_label = paper_period_label(observation.get("period_number", 1))
        payoff_block = paper_payoff_block(self.name)
        reason_instruction = observation["reason_instruction"]
        reason_field = ""
        if reason_instruction != "optional empty string":
            reason_field = f',\n            "reason": "{reason_instruction}"'

        worked_example = ""
        if observation.get("prompt_modifier") != "remove_worked_examples":
            worked_example = """
            Example 6 Suppose you choose the “I will contribute 5 Taler if in total at least 20 Taler are contributed (by all
            members of the group including yourself)” Suppose the other participants contribute 18 Taler in total. Together
            with your 5 Taler it would be 18 + 5 = 23 Taler. Your condition would be fulfilled and you would contribute
            5 Taler.
            """
        worked_interpretation = ""
        if observation.get("prompt_modifier") != "remove_worked_examples":
            worked_interpretation = """
            In example 6 you would contribute 5 Taler if the others in your group contribute at least 15 such that the
            condition of 20 is reached. Otherwise you would not contribute.
            """
        return f"""{prompt}

            Each participant can condition his contribution to the common project on how much all other participants
            contribute. All conditions have the following form:

            “I will contribute X Taler, if in total at least Y Taler are contributed (by all members of the group including
            yourself)”

            {worked_example}

            The computer then chooses the highest total amount for the project that is compatible with the conditions such
            that only those participants contribute whose conditions are satisfied.

            {worked_interpretation}

            {payoff_block}

            Structure of the screen

            You have a printed example of the structure of the screen of the program you will make your decisions in each
            period. The screen is divided into three blocks.

            The upper left block contains a calculator. Here you can test contributions and conditions for yourself and the
            other four participants (by default, the computer enters the numbers group members chose in the previous period.
            You can change them, of course). As soon as you have chosen contributions and conditions for all participants you
            can press the button “Calculate payoff”. Then the computer calculates the payoff you will receive in this case and
            the amount you would invest.

            In the upper right block you enter your contributions and conditions that will be relevant for your later payoff
            (again, by default, the computer enters the numbers you chose in the previous period). If you want to change them,
            do it now. Beneath that, there is a red button. If you press it you submit your decision and leave the screen. Only
            when all participants have pressed the red button the experiment will continue. A time display on the right gives
            you a hint until when you should have made a decision. If the time runs out this has no consequences.

            In the bottom block you will see from the second period onwards which contributions and conditions all
            participants chose in the previous period. In the first period this block will be empty.

            {period_label}

            Information displayed on the current screen:

            {history_text}

            Return ONLY valid JSON with this schema:
            {{
            "contribution_1": integer from 0 to 10,
            "threshold_1": integer from 0 to 50{reason_field}
            }}
        """

    def resolve(self, decisions, context):
        message_pairs = [[(decision.contribution_1, decision.threshold_1)] for decision in decisions]
        contributions, total = ccm_resolve(message_pairs)
        return {
            "rows": [
                {
                    "contribution_1": decision.contribution_1,
                    "threshold_1": decision.threshold_1,
                    "implemented_contribution": contribution,
                    "contribution_share": contribution_share(contribution),
                    "contributes": contribution > 0,
                    "total_contribution": total,
                    "total_contributors": sum(c > 0 for c in contributions),
                    "payoff": public_good_payoff(contribution, total),
                }
                for decision, contribution in zip(decisions, contributions)
            ],
            "contributors": sum(c > 0 for c in contributions),
            "total_contribution": total,
        }
