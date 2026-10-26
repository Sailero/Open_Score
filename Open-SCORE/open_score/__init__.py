"""Current HAD experiments; native environment imports are deferred until use."""
__version__ = "6.4.0"
__all__ = ["make_env", "parallel_env", "__version__"]

def __getattr__(name):
    if name in ("make_env", "parallel_env"):
        from . import envs
        value = getattr(envs, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
