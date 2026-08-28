from src.games.base import Game
from src.reporting import common_row


RO_PLAYERS = 5
RO_PERIODS = 60
RO_ENDOWMENT = 10
RO_MIN_CONTRIBUTION = 0
RO_MAX_CONTRIBUTION = 10
RO_MPCR = 0.4


_PAPER_INTRO_CCM_SCCM = """
Welcome to our experiment! Please read these instructions carefully. Do not talk to your neighbor from now on.
Please turn off your mobile phone and leave it turned off until the end of the experiment. If you have any
questions, raise your hand. We will come to you. All participants have the same instructions.

In the experiment you will be divided in groups of 5. The experiment will last for 60 periods. In each period
you will interact with the same 4 other participants. The experiment runs completely anonymously. No participant
will be informed about who he has interacted with or which payoff another participant received.

The currency for the experiment will be Taler. In each period you will start with 10 Taler. In each period you
can decide which (integer valued) fraction of those 10 Taler you want to invest into a common project. For every
Taler invested by participants of your group into the common project, each member of the group will be paid 0.4
Taler. You will keep the rest of the 10 Taler.

Example 1 You invest 4 Taler into the project and additionally 2 other participants each invested 8 Taler, thus
in total 20 (= 4+8+8) Taler are invested. You will receive for your investment and the investments of the two
other participants in total 20 × 0.4 = 8 Taler. You keep the 6 Taler you did not invest. So you earn in total
8 + 6 = 14 Taler in this period.

Example 2 All participants invest 10 Taler into the common project. So you earn 50 × 0.4 = 20 Taler in this
period.

Example 3 All participants invest 0 Taler into the common project. So you simply keep the 10 Taler you received
in the beginning.

Example 4 All the other participants invest 10 Taler into the common project, you invest nothing. So you earn
40 × 0.4 + 10 = 26 Taler in this period.

Example 5 All the other participants invest nothing into the common project, you invest 10 Taler. So you earn
10 × 0.4 = 4 Taler in this period.
""".strip()


_PAPER_INTRO_VCM = """
Welcome to our experiment! Please read the instructions carefully. Do not talk to your neighbor from now on.
Please turn off your mobile phone and leave it turned off until the end of the experiment. If you have any
questions, raise your hands. We will come to you. All participants have the same instructions.

In the experiment you will be divided in groups of 5. The experiment will last for 60 periods. You will be
grouped with the same four participants in all periods. In each period you will interact with the same 4 other
participants. The experiment runs completely anonymously. No participant will be informed about who he has
interacted with or which payoff another participant received.

The currency for the experiment will be Taler. In each period you will start with 10 Taler. In each period you
can decide which (integer valued) fraction of those 10 Taler you want to invest into a common project. For every
Taler invested by participants of your group into the common project, each member of the group will be paid 0.4
Taler. You will keep the rest of the 10 Taler.

Example 1 You invest 4 Taler into the project and additionally 2 other participants each invested 8 Taler, thus
in total 20 (= 4+8+8) Taler are invested. You will receive for your investment and the investments of the two
other participants in total 20 × 0.4 = 8 Taler. You keep the 6 Taler you did not invest. So you earn in total
8 + 6 = 14 Taler in this period.

Example 2 All participants invest their 10 Taler into the common project. So you earn 50 × 0.4 = 20 Taler in
this period.

Example 3 All participants invest 0 Taler into the common project. So you simply keep the 10 Taler you received
in the beginning.

Example 4 All the other participants invest 10 Taler into the common project, you invest nothing. So you earn
40 × 0.4 + 10 = 26 Taler in this period.

Example 5 All the other participants invest nothing into the common project, you invest 10 Taler. So you earn
10 × 0.4 = 4 Taler in this period.
""".strip()


_PAPER_PAYOFF_CCM_SCCM = """
Payoff

The experiment lasts for 60 periods. A questionnaire will follow after the 60 periods. At the end of the
experiment one of the 60 periods will be randomly chosen and the earnings of the chosen period will be paid with
an exchange factor of 1 Taler = 1 €. The payment will be private and in cash. For example, if your earnings from
the payment relevant period are 15 Taler, you will receive 15 €.
""".strip()


_PAPER_PAYOFF_VCM = """
Payoff

The experiment lasts for 60 periods. A questionnaire will follow after the 60 periods. At the end of the
experiment one of the 60 periods will be randomly chosen and the earnings of the chosen period will be paid with
an exchange factor of 1 Taler = 1 €. The payment will be private and in cash. For example, if your earnings from
the payment relevant period is 15 Taler, you will receive 15 €.
""".strip()


_GENERIC_GAME_INTRO = """
Welcome to our experiment.

In the experiment you are in a group of 5 participants. The experiment lasts for 60 periods. In every period you
interact with the same 4 other participants. The experiment is completely anonymous. No participant is informed
about the identities or point payoffs of the other participants.

The currency is Taler. In every period you start with 10 Taler. You can invest any integer amount from 0 to 10
Taler into a common project and keep the remainder. For every Taler invested by any participant in your group,
every group member receives 0.4 Taler.
""".strip()


def _canonical_mechanism_name(mechanism_name: str | None) -> str:
    name = str(mechanism_name or "")
    return name[3:] if name.startswith("ro_") else name


def paper_game_intro(mechanism_name: str | None) -> str:
    mechanism = _canonical_mechanism_name(mechanism_name)
    if mechanism == "vcm":
        # Appendix D contains the clear typo "50 × 0.64 = 20". We use 0.4,
        # which is both the stated MPCR and the only value consistent with 20.
        return _PAPER_INTRO_VCM
    if mechanism in {"sccm", "ccm"}:
        return _PAPER_INTRO_CCM_SCCM
    return _GENERIC_GAME_INTRO


def paper_payoff_block(mechanism_name: str | None) -> str:
    mechanism = _canonical_mechanism_name(mechanism_name)
    return _PAPER_PAYOFF_VCM if mechanism == "vcm" else _PAPER_PAYOFF_CCM_SCCM


def paper_period_label(period_number: int) -> str:
    return f"Period {period_number} out of {RO_PERIODS}."


def contribution_share(contribution_units: float) -> float:
    return contribution_units / RO_ENDOWMENT


def public_good_payoff(contribution_units: float, total_contribution_units: float) -> float:
    return RO_ENDOWMENT - contribution_units + RO_MPCR * total_contribution_units


class ROPublicGoodGame(Game):
    """Complete-information public-good game from Oechssler et al. (2022)."""

    name = "ro_public_good"

    def observation(self, context):
        if context["players"] != RO_PLAYERS:
            raise ValueError(
                f"The Oechssler et al. (2022) reproduction requires exactly {RO_PLAYERS} players; "
                f"received {context['players']}."
            )
        return {
            **context["meta"],
            "treatment": context["treatment"],
            "chain_id": context["chain"],
            "generation": context["generation"],
            "period_number": context["generation"],
            "players_in_generation": context["players"],
            "repetition": context["repetition"],
            "effective_seed": None,
            "history": context["history"],
        }

    def prompt(self, observation, reason_instruction):
        del reason_instruction
        return paper_game_intro(observation.get("mechanism"))

    @staticmethod
    def _recorded_choice(decision):
        return decision.condition if decision.condition is not None else decision.choice

    def rows(self, context, decisions, outcome):
        rows = []
        for player, (decision, mechanism_row) in enumerate(zip(decisions, outcome["rows"]), 1):
            row = common_row(context, player, decision) | {
                "choice": self._recorded_choice(decision),
                "player_type": "fixed" if player in context.get("fixed_player_ids", set()) else "llm",
                "fixed_policy": context.get("fixed_policy", "none"),
                "fixed_count": context.get("fixed_count", 0),
                "fixed_placement": context.get("fixed_placement", "lowest_player_ids"),
                **mechanism_row,
            }
            rows.append(row)

        history_rows = [
            {
                "player": row["player"],
                "choice": row["choice"],
                "condition": row.get("condition", row["choice"]),
                "contributes": row["contributes"],
                "implemented_contribution": row.get("implemented_contribution", row.get("choice", 0)),
                "total_contribution": row.get("total_contribution"),
                "contribution_1": row.get("contribution_1"),
                "threshold_1": row.get("threshold_1"),
                "contribution_2": row.get("contribution_2"),
                "threshold_2": row.get("threshold_2"),
                "payoff": row["payoff"],
            }
            for row in rows
        ]
        return rows, {"history_entry": history_rows}
