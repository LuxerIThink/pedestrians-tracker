import sys
import pytest
from io import StringIO
import test_main
from unittest.mock import patch
from main import get_dataset_path


def test_get_dataset_path_with_argument():
    file_path = "test_dataset"
    with patch.object(sys, 'argv', ['', file_path]):
        assert get_dataset_path() == file_path


def test_get_dataset_path_without_argument(capsys):
    with patch.object(sys, 'argv', ['']):
        with pytest.raises(SystemExit) as exc:
            get_dataset_path()
        assert "Add path to dataset as first argument!" in capsys.readouterr().out
        assert exc.type == SystemExit
        assert exc.value.code == 1