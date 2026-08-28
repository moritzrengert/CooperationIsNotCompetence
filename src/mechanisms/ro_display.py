from __future__ import annotations


def rows_in_player_view(rows: list[dict], player: int | None) -> list[dict]:
    """Order one period as on the laboratory screen: self first, then fixed peers."""
    if player is None:
        return list(rows)

    own_rows = [row for row in rows if row.get("player") == player]
    other_rows = sorted(
        (row for row in rows if row.get("player") != player),
        key=lambda row: int(row.get("player", 0)),
    )
    if len(own_rows) != 1:
        raise ValueError(f"Expected exactly one row for player {player}; found {len(own_rows)}.")
    return own_rows + other_rows


def previous_period_for_prompt(history: list[list[dict]]) -> tuple[int, list[dict]] | None:
    """Return exactly the information displayed by the paper's current screen.

    Human participants could remember earlier periods internally, but the interface
    explicitly displayed only the immediately preceding period. The strict paper
    reproduction therefore does not inject older rounds into the prompt.
    """
    if not history:
        return None
    return len(history), history[-1]
