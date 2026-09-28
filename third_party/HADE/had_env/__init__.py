"""Standalone HAD environments and optional research visualization."""

__version__ = "1.0.0"


def make_env(*args, **kwargs):
    """Create a named HAD scenario; see :func:`had_env.factory.make_env`."""
    from .factory import make_env as create
    return create(*args, **kwargs)


def parallel_env(**kwargs):
    """Construct the native simultaneous defense environment."""
    from .factory import parallel_env as create
    return create(**kwargs)


__all__ = ["make_env", "parallel_env", "__version__"]
