from printerfail.io.paths import get_dir_from_env
import os
from pathlib import Path

def test_get_dir_from_env(tmp_path):
    os.environ["X_TEST_DIR"] = str(tmp_path / "abc")
    p = get_dir_from_env("X_TEST_DIR", "./fallback")
    assert isinstance(p, Path)
    # Resolve-olt path jön vissza
    assert str(p).endswith("abc")
