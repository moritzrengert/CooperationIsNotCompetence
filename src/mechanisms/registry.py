from src.mechanisms.bccm import BCCMMechanism
from src.mechanisms.ccm import CCMMechanism
from src.mechanisms.ccf import CCFMechanism
from src.mechanisms.fischer_baseline import FischerBaselineMechanism, NoMechanism
from src.mechanisms.sccm import SCCMMechanism
from src.mechanisms.vcm import VCMMechanism


_BCCM = BCCMMechanism()
_CCM = CCMMechanism()
_CCF = CCFMechanism()
_SCCM = SCCMMechanism()
_VCM = VCMMechanism()

_MECHANISMS = {
    "none": NoMechanism(),
    "fischer_2004": FischerBaselineMechanism(),
    "bccm": _BCCM,
    "ccm": _CCM,
    "ccf": _CCF,
    "sccm": _SCCM,
    "vcm": _VCM,
    # Backward-compatible aliases for older internal references.
    "ro_bccm": _BCCM,
    "ro_ccm": _CCM,
    "ro_ccf": _CCF,
    "ro_sccm": _SCCM,
    "ro_vcm": _VCM,
}


def get_mechanism(name: str):
    if name not in _MECHANISMS:
        raise ValueError(f"Unknown mechanism: {name}. Available: {sorted(_MECHANISMS)}")
    return _MECHANISMS[name]
