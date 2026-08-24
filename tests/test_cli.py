import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from lumbar_stenosis_ai.cli import main


class CliTests(unittest.TestCase):
    def test_self_check(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["self-check"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["synthetic_only"])
        self.assertIn("RESEARCH PROTOTYPE", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
