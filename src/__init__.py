__all__ = ["run_pipeline"]


def run_pipeline(*args, **kwargs):
    """Run the ML pipeline, importing orchestration dependencies lazily."""
    from .orchestration import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
