import io
import json
import unittest
from contextlib import redirect_stdout

from machboost import __version__
from machboost.cli import doctor_data, main, self_test_data


class CLITests(unittest.TestCase):
    def test_doctor_data_has_optional_package_statuses(self):
        data = doctor_data()

        self.assertEqual(data["schema_version"], "machboost.doctor.v1")
        self.assertEqual(data["machboost_version"], __version__)
        self.assertIn("transformers", data["optional_packages"])
        self.assertIn("available", data["optional_packages"]["transformers"])

    def test_self_test_uses_verifier_path(self):
        data = self_test_data()

        self.assertTrue(data["ok"])
        self.assertTrue(data["output_match"])
        self.assertGreater(data["accepted_draft_tokens"], 0)
        self.assertGreater(data["estimated_speedup"], 1.0)

    def test_main_version_prints_version(self):
        output = io.StringIO()

        with redirect_stdout(output):
            code = main(["version"])

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue().strip(), __version__)

    def test_main_self_test_json(self):
        output = io.StringIO()

        with redirect_stdout(output):
            code = main(["self-test", "--json"])

        data = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(data["ok"])


if __name__ == "__main__":
    unittest.main()
