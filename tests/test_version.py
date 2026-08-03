import json
import tomllib
import unittest
from pathlib import Path

import machboost


ROOT = Path(__file__).resolve().parents[1]


class VersionConsistencyTests(unittest.TestCase):
    def test_package_and_desktop_runtime_versions_match(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        manifest = json.loads(
            (ROOT / "apps/macos/Resources/RuntimeManifest.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(machboost.__version__, project["project"]["version"])
        self.assertEqual(
            machboost.__version__,
            manifest["packages"]["machboost"],
        )


if __name__ == "__main__":
    unittest.main()
