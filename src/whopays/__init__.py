"""WhoPays: per-language audit of inference optimizations. ParityQuant: worst-language bits."""

__version__ = "0.1.0"


def run(jobs, **kwargs):
    """Run a job list. See `whopays.run.run`; imported lazily so `import whopays` stays cheap."""
    from whopays.run import run as _run

    return _run(jobs, **kwargs)
