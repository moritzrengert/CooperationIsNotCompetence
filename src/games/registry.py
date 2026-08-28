from src.games.fischer_2004 import Fischer2004Game
from src.games.ro_public_good import ROPublicGoodGame


_RO_PUBLIC_GOOD = ROPublicGoodGame()

_GAMES = {
    "fischer_2004": Fischer2004Game(),
    "ro_public_good": _RO_PUBLIC_GOOD,
    # Backward-compatible aliases for older configs and mechanism-specific runs.
    "ro_bccm": _RO_PUBLIC_GOOD,
    "ro_ccm": _RO_PUBLIC_GOOD,
    "ro_ccf": _RO_PUBLIC_GOOD,
    "ro_sccm": _RO_PUBLIC_GOOD,
    "ro_vcm": _RO_PUBLIC_GOOD,
}


def get_game(name: str):
    if name not in _GAMES:
        raise ValueError(f"Unknown game: {name}. Available: {sorted(_GAMES)}")
    return _GAMES[name]
