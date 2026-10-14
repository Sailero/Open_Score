"""Load the selected runner without importing Torch in environment workers."""
def parallel_runner(*args, **kwargs):
    from .parallel_runner import ParallelRunner
    return ParallelRunner(*args, **kwargs)


REGISTRY = {"parallel": parallel_runner}
