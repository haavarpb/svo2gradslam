from importlib.resources import files
from pathlib import Path


def test_svo_exists():
    path = files("svo2gradslam").joinpath("sofa.svo")
    assert Path(path).exists()
