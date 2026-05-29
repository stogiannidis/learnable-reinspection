from unittest.mock import MagicMock, patch

from src.config import ReInspectionConfig
from src.utils.progress import make_eval_tqdm


def test_make_eval_tqdm_uses_stdlib_when_disabled():
    config = ReInspectionConfig(pbar_enabled=False)
    with patch("src.utils.progress.std_tqdm", create=True) as mock_std:
        mock_std.return_value = MagicMock()
        with patch("tqdm.tqdm", mock_std):
            make_eval_tqdm(config, range(3), desc="test", disable=False)
    mock_std.assert_called_once()
    args, kwargs = mock_std.call_args
    assert kwargs["desc"] == "test"
    assert kwargs["disable"] is False


def test_make_eval_tqdm_keeps_cloud_bar_enabled_under_tee():
    # Under tee (disable=True) the bar must stay ENABLED so pbar.io keeps syncing;
    # local rendering is redirected to a devnull file and miniters is made fine-grained.
    config = ReInspectionConfig(pbar_enabled=True)
    cloud_tqdm = MagicMock(return_value=MagicMock())
    with patch("src.utils.progress.get_tqdm", return_value=cloud_tqdm) as mock_get:
        make_eval_tqdm(config, range(3), desc="frozen/vsr", disable=True, miniters=200)
    mock_get.assert_called_once_with(config, cloud=True)
    args, kwargs = cloud_tqdm.call_args
    assert args == (range(3),)
    assert kwargs["desc"] == "frozen/vsr"
    assert kwargs["disable"] is False
    assert kwargs["miniters"] == 1
    assert kwargs["file"] is not None


def test_make_eval_tqdm_uses_cloud_tqdm_when_display_enabled():
    config = ReInspectionConfig(pbar_enabled=True)
    cloud_tqdm = MagicMock(return_value=MagicMock())
    with patch("src.utils.progress.get_tqdm", return_value=cloud_tqdm) as mock_get:
        make_eval_tqdm(config, range(5), desc="reinspection/gqa", disable=False, miniters=10)
    mock_get.assert_called_once_with(config, cloud=True)
    cloud_tqdm.assert_called_once_with(
        range(5),
        desc="reinspection/gqa",
        disable=False,
        miniters=10,
    )
