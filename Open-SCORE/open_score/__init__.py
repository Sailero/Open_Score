"""Cross-scale damage-defense experiments backed by the HAD Workbench."""
def __getattr__(name):
    # Read-only CLI/status utilities need no simulator installation. The
    # existing environment factories retain their public names and objects.
    if name in ("make_env", "parallel_env"):
        from . import envs
        value = getattr(envs, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "6.4.0"
__all__ = ["make_env", "parallel_env", "__version__"]
