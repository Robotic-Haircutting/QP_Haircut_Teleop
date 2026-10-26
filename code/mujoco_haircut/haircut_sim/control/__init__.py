"""Local robot control, lazy imports."""
from importlib import import_module
_EXPORTS = {
 'PiperFollower': '.piper_follower', 'PiperFollowerConfig': '.config_piper_follower',
 'SO101LeaderEE': '.so101_leader_ee', 'SO101LeaderEEConfig': '.config_so101_leader_ee',
 'SO101LeaderHome': '.so101_leader_home', 'SO101LeaderHomeConfig': '.config_so101_leader_home'}
__all__ = list(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
def register_devices():
    for name in __all__:
        __getattr__(name)
