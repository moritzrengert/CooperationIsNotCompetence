from src.games.base import Game
from src.games.treatments import Treatment
from src.reporting import common_row


def production(total_effort: int) -> float:
    if total_effort <= 9:
        return 0.6 * total_effort
    return 8.1 - 0.3 * total_effort


def payoff_share(own_effort: int, total_effort: int) -> float:
    if total_effort <= 0:
        return 0.0
    return own_effort / total_effort * production(total_effort)


def payoff(own_effort: int, other_effort_sum: int, reserve: float) -> float:
    total = own_effort + other_effort_sum
    return payoff_share(own_effort, total) * reserve


def prediction_bonus(prediction: int, other_effort_sum: int) -> int:
    """Paper's prediction payment: 20 Taler minus 1 per unit error, floored at 0."""
    return max(0, 20 - abs(prediction - other_effort_sum))


def next_reserve(reserve: float, total_effort: int, treatment: Treatment) -> float:
    if treatment == Treatment.RESTART:
        return reserve
    if treatment == Treatment.FAST:
        return (1 - (total_effort - 21) / 24) * reserve
    if treatment == Treatment.SLOW:
        return (1 - (total_effort - 6) / 24) * reserve
    raise ValueError(f"Unknown treatment: {treatment}")


def best_reply(predicted_other_sum: int) -> int:
    return max(range(1, 9), key=lambda own: payoff_share(own, own + predicted_other_sum))


def round_reserve(value: float) -> int:
    # The paper states that reserves were rounded to the next integer.  Use
    # conventional half-up rounding instead of Python's banker's rounding.
    return int(value + 0.5)


class Fischer2004Game(Game):
    name = "fischer_2004"

    def observation(self, context):
        return {
            **context["meta"],
            "treatment": context["treatment"],
            "chain_id": context["chain"],
            "generation": context["generation"],
            "players_in_generation": context["players"],
            "current_reserve": context["reserve"],
            "choice_options": list(range(1, 9)),
            "prediction_options": list(range(2, 17)),
            "repetition": context["repetition"],
            "effective_seed": None,
        }

    def prompt(self, observation, reason_instruction):
        prompt_style = observation.get("prompt_style", "minimal")
        if prompt_style == "human_table":
            return self._human_table_prompt(observation, reason_instruction)
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    def rows(self, context, decisions, outcome):
        efforts = outcome["implemented_choices"]
        contributors = outcome.get("contributors")
        total = sum(efforts)
        rows = []

        for player, (decision, effort) in enumerate(zip(decisions, efforts), 1):
            other_sum = total - effort
            bonus = prediction_bonus(decision.prediction, other_sum)
            game_payoff = payoff(effort, other_sum, context["reserve"])
            row = common_row(context, player, decision) | {
                "reserve": context["reserve"],
                "choice": decision.choice,
                "implemented_choice": effort,
                "prediction": decision.prediction,
                "prediction_bonus": bonus,
                "best_reply_to_prediction": best_reply(decision.prediction),
                "other_actual_sum": other_sum,
                "payoff": game_payoff,
                "total_payoff": game_payoff + bonus,
            }
            if contributors is not None:
                activated = decision.condition <= contributors
                row |= {
                    "condition": decision.condition,
                    "offer_activated": activated,
                    "total_activated_offers": contributors,
                    "contributes": activated,
                    "total_contributors": contributors,
                }
            rows.append(row)

        next_state = {
            "reserve": round_reserve(next_reserve(context["reserve"], total, Treatment(context["treatment"])))
        }
        return rows, next_state

    @staticmethod
    def _position_information_text(observation):
        generation = observation.get("generation")
        total_generations = observation.get("total_generations")
        if generation is None or total_generations is None:
            return "You are not informed about the position of your own group within the chain."
        return f"You are informed that your group is generation {generation} out of {total_generations} in the chain."

    @staticmethod
    def _human_table_prompt(observation, reason_instruction):
        decision_table = Fischer2004Game._decision_table(observation["treatment"])

        return f"""
            Welcome to this experiment!

            You are one participant in a group of three. A number of these groups form a chain.
            A chain consists of a first group, a last group, and an undisclosed number of intermediate groups.
            {Fischer2004Game._position_information_text(observation)}

            An endowment is made available to the first group in the chain. Every other group receives the
            endowment that the preceding group has left over. Thus, the endowment is passed from one group
            to the next and develops according to the decisions in the chain. The payoff potential of a group
            depends on the endowment left to it. Apart from possible differences in endowment, the decision
            situation is identical for all participants in a chain.

            The other two members of your group make their decisions simultaneously. You do not observe their
            decisions before making your own decision. The participants currently visible to you are associated
            with other chains, not with your group.

            Decisions and payoffs in different chains are completely independent from each other.

            Your current endowment is: {observation["current_reserve"]}

            Your task is to choose one of the numbers 1, 2, 3, 4, 5, 6, 7, or 8.

            The decisions in a group jointly affect the payoffs of the group members. The table below shows
            these effects. Rows are your possible choices. Columns are the sum of the choices of the other two
            members of your group.

            Each table cell has the format:

            payoff_percent / endowment_change_percent

            The payoff percent is the percentage of the current endowment that you receive as your payoff.
            The endowment change percent is the percentage by which the current endowment is increased or
            decreased for the succeeding group in the chain. A value of +0.0% means that the endowment is
            unchanged.

            {decision_table}

            Furthermore, it is your task to predict the sum of the decisions of the other two members of your group.
            The prediction must be an integer from 2 to 16. For a perfect prediction you receive an additional
            20 Taler; one Taler is deducted for each unit by which your prediction differs from the actual sum.

            Return ONLY valid JSON with this schema:

            {{
            "choice": integer from 1 to 8,
            "prediction": integer from 2 to 16,
            "reason": "{reason_instruction}"
            }}
            """

    @staticmethod
    def _decision_table(treatment_value):
        treatment = Treatment(treatment_value)
        reserve = 100.0
        columns = list(range(2, 17))
        rows = [
            "| Own choice | " + " | ".join(str(col) for col in columns) + " |",
            "|---|" + "|".join("---" for _ in columns) + "|",
        ]

        for own_choice in range(1, 9):
            cells = []
            for other_sum in columns:
                total_effort = own_choice + other_sum
                own_payoff_percent = payoff_share(own_choice, total_effort) * 100
                updated_reserve = next_reserve(reserve, total_effort, treatment)
                endowment_change_percent = ((updated_reserve / reserve) - 1) * 100
                cells.append(f"{own_payoff_percent:.1f}% / {endowment_change_percent:+.1f}%")

            rows.append(f"| {own_choice} | " + " | ".join(cells) + " |")

        return "\n".join(rows)
