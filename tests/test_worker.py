import unittest
from unittest.mock import patch

from imprint.worker import watch_parent


class ParentWatchTests(unittest.TestCase):
    def test_parent_loss_exits_only_current_worker(self):
        class Stop:
            def wait(self, timeout):
                return False

        with (
            patch("imprint.worker.os.getppid", return_value=999),
            patch(
                "imprint.worker.os._exit", side_effect=SystemExit(0)
            ) as exit_worker,
        ):
            with self.assertRaises(SystemExit):
                watch_parent(123, Stop())
            exit_worker.assert_called_once_with(0)

    def test_normal_shutdown_does_not_exit_process(self):
        class Stop:
            def wait(self, timeout):
                return True

        with patch("imprint.worker.os._exit") as exit_worker:
            watch_parent(123, Stop())
            exit_worker.assert_not_called()
