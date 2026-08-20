import importlib.util
from pathlib import Path


TRAINER_PATH = Path(__file__).resolve().parents[1] / "src" / "training" / "trainer.py"
spec = importlib.util.spec_from_file_location("trainer", TRAINER_PATH)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_build_model_uses_configured_random_state():
    estimator, _ = trainer._build_model({"random_state": 123, "n_jobs": 1})

    assert estimator.random_state == 123
    assert estimator.n_jobs == 1


def test_build_model_defaults_random_state_for_reproducibility():
    estimator, _ = trainer._build_model({})

    assert estimator.random_state == 42
