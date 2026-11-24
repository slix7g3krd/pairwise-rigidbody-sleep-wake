import pytest
import gameplay.rigidbody_sleep_wake as mod

CASES = [
    ("sleep_hysteresis", (1,)),
    ("wake_connected", (1,)),
    ("wake_immediate", (1,)),
    ("refresh_all_transforms", (1,)),
    ("register_for_unload", (1,)),
    ("clear_on_rebuild", (1,)),
    ("simulate", (1,)),
    ("sleep", (1,)),
    ("wake", (1,)),
    ("unload", (1,)),
    ("validate_sleep", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
