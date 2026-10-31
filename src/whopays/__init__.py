"""WhoPays: per-language audit of inference optimizations. ParityQuant: worst-language bits."""

__version__ = "0.1.0"

# Import once: a lazy wrapper gets overwritten by Python's same-named submodule,
# making whopays.run(...) depend on import order.
from whopays.run import run as run  # noqa: E402
