"""Load the selected runner without importing Torch in environment workers."""
def episode_runner(*args, **kwargs):
    from .episode_runner import EpisodeRunner
    return EpisodeRunner(*args, **kwargs)


def parallel_runner(*args, **kwargs):
    from .parallel_runner import ParallelRunner
    return ParallelRunner(*args, **kwargs)


REGISTRY = {"episode": episode_runner, "parallel": parallel_runner}
