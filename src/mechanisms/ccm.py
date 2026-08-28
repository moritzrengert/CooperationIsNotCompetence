from src.games.ro_public_good import contribution_share, paper_payoff_block, paper_period_label, public_good_payoff
from src.mechanisms.base import Mechanism
from src.mechanisms.ro_display import previous_period_for_prompt, rows_in_player_view


def _offer_value(total: int, offers: list[tuple[int, int]]) -> int:
    return max((contribution for contribution, threshold in offers if total >= threshold), default=0)


def ccm_resolve(message_pairs: list[list[tuple[int, int]]]) -> tuple[list[int], int]:
    """Resolve Oechssler et al. (2022) CCM messages on an integer 0..10 grid."""
    max_total = 10 * len(message_pairs)
    total = max(t for t in range(max_total + 1) if sum(_offer_value(t, offers) for offers in message_pairs) >= t)
    contributions = [_offer_value(total, offers) for offers in message_pairs]
    realized_total = sum(contributions)
    if realized_total != total:
        total = realized_total
        contributions = [_offer_value(total, offers) for offers in message_pairs]
    return contributions, total


def scaled_public_good_payoff(contribution_units: int, total_units: int) -> float:
    return public_good_payoff(contribution_units, total_units)


def _format_history(history: list[list[dict]], player: int | None) -> str:
    """Render exactly the previous-period information shown by the CCM screen."""
    previous = previous_period_for_prompt(history)
    if previous is None:
        return "In the first period there is no previous history and the bottom block is empty."

    previous_period, rows = previous
    ordered = rows_in_player_view(rows, player)
    messages = [
        (
            row.get("contribution_1"),
            row.get("threshold_1"),
            row.get("contribution_2"),
            row.get("threshold_2"),
        )
        for row in ordered
    ]
    implemented = [row["implemented_contribution"] for row in ordered]
    own_row = ordered[0]
    return (
        f"Previous-period block (period {previous_period}): conditions (X1, Y1, X2, Y2) for Player 1 "
        f"(Yourself), Player 2, Player 3, Player 4, and Player 5 = {messages}; implemented contributions "
        f"in the same order = {implemented}.\n"
        "Defaults on the current screen: your two condition fields are prefilled with your own previous "
        f"conditions ({own_row.get('contribution_1')}, {own_row.get('threshold_1')}) and "
        f"({own_row.get('contribution_2')}, {own_row.get('threshold_2')}). The calculator is prefilled "
        f"with the previous period's ten conditions: {messages}. You may keep or change any prefilled value."
    )


class CCMMechanism(Mechanism):
    """Conditional Contribution Mechanism from Oechssler et al. (2022)."""

    name = "ccm"

    def augment_observation(self, observation, context):
        observation = observation.copy()
        observation["contribution_options"] = list(range(11))
        observation["threshold_options"] = list(range(10 * context["players"] + 1))
        observation["offer_count"] = 2
        return observation

    def augment_prompt(self, prompt, observation):
        # The shared game block comes from the wording common to Appendices B/F.
        # This file adds Appendix B's CCM-specific rules, payoff/screen material,
        # and the period, previous-screen state, and machine-readable response block.
        history_text = _format_history(observation.get("history", []), observation.get("player"))
        period_label = paper_period_label(observation.get("period_number", 1))
        payoff_block = paper_payoff_block(self.name)
        reason_instruction = observation["reason_instruction"]
        reason_field = ""
        if reason_instruction != "optional empty string":
            reason_field = f',\n            "reason": "{reason_instruction}"'

        prompt_modifier = observation.get("prompt_modifier")
        worked_examples = ""
        if prompt_modifier == "coalition_compatible_worked_examples":
            worked_examples = """
            Example 6 Suppose you choose the “I will contribute 5 Taler if in total at least 15 Taler are contributed (by all
            members of the group including yourself)” Suppose the other participants contribute 10 Taler in total. Together
            with your 5 Taler it would make 10 + 5 = 15 Taler. Your condition would be fulfilled and you would contribute
            5 Taler.

            In fact, you should formulate two different conditions of this kind. For example, you could signal the other
            participants that you would be willing to contribute more, if enough of the others do so too. E.g. we could append
            the condition from example 6 by a higher condition:

            First condition: “I will contribute 5 Taler, if in total at least 15 Taler are contributed (by all members of the
            group including yourself)”

            Second condition: “I will contribute 10 Taler, if in total at least 30 Taler are contributed (by all members of
            the group including yourself)”
            """
        elif prompt_modifier != "remove_worked_examples":
            worked_examples = """
            Example 6 Suppose you choose the “I will contribute 5 Taler if in total at least 20 Taler are contributed (by all
            members of the group including yourself)” Suppose the other participants contribute 18 Taler in total. Together
            with your 5 Taler it would make 18 + 5 = 23 Taler. Your condition would be fulfilled and you would contribute
            5 Taler.

            In fact, you should formulate two different conditions of this kind. For example, you could signal the other
            participants that you would be willing to contribute more, if all the others do so too. E.g. we could append the
            condition from example 6 by a higher condition:

            First condition: “I will contribute 5 Taler, if in total at least 20 Taler are contributed (by all members of the
            group including yourself)”

            Second condition: “I will contribute 10 Taler, if in total at least 50 Taler are contributed (by all members of
            the group including yourself)”
            """
        worked_consequences = ""
        if prompt_modifier == "coalition_compatible_worked_examples":
            worked_consequences = """
            For example, if you choose the two conditions from above, you will contribute 10 Taler if the group contributes
            in total at least 30 Taler. If the group contributes in total an amount between 15 and 29 Taler you will contribute
            5 Taler. And if the others in the group contribute less than 10 Taler such that even with your 5 Taler the
            condition of 15 Taler will not be reached, you will contribute nothing.
            """
        elif prompt_modifier != "remove_worked_examples":
            worked_consequences = """
            For example, if you choose the two conditions from above, you will contribute 10 Taler if also all other
            participants contribute the maximal amount of 10 Taler, such that in total 50 Taler are contributed. If the group
            contributes in total an amount between 20 and 49 Taler you will contribute 5 Taler. And if the others in the group
            contribute less than 15 Taler such that even with your 5 Taler the condition of 20 Taler will not be reached, you
            will contribute nothing.
            """
        condition_requirement = ""
        if prompt_modifier == "remove_worked_examples":
            condition_requirement = "In fact, you should formulate two different conditions of this kind."
        return f"""{prompt}

            Each participant can condition his contribution to the common project on how much all other participants
            contribute. All conditions have the following form:

            “I will contribute X Taler, if in total at least Y Taler are contributed (by all members of the group including
            yourself)”

            {worked_examples}

            {condition_requirement}

            The computer then chooses the condition for each participant, such that the highest total amount for the project
            is reached. But the conditions make sure that you never pay more than you indicated.

            {worked_consequences}

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
            "threshold_1": integer from 0 to 50,
            "contribution_2": integer from 0 to 10,
            "threshold_2": integer from 0 to 50{reason_field}
            }}
        """

    def resolve(self, decisions, context):
        message_pairs = [
            [
                (decision.contribution_1, decision.threshold_1),
                (decision.contribution_2, decision.threshold_2),
            ]
            for decision in decisions
        ]
        contributions, total = ccm_resolve(message_pairs)
        return {
            "rows": [
                {
                    "contribution_1": decision.contribution_1,
                    "threshold_1": decision.threshold_1,
                    "contribution_2": decision.contribution_2,
                    "threshold_2": decision.threshold_2,
                    "implemented_contribution": contribution,
                    "contribution_share": contribution_share(contribution),
                    "contributes": contribution > 0,
                    "total_contribution": total,
                    "total_contributors": sum(c > 0 for c in contributions),
                    "payoff": scaled_public_good_payoff(contribution, total),
                }
                for decision, contribution in zip(decisions, contributions)
            ],
            "contributors": sum(c > 0 for c in contributions),
            "total_contribution": total,
        }
