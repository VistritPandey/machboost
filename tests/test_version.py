import json
import re
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

    def test_desktop_build_number_increases_with_semantic_version(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        major, minor, patch = (
            int(part) for part in project["project"]["version"].split(".")
        )
        expected = major * 1_000_000 + minor * 1_000 + patch
        xcodegen = (ROOT / "apps/macos/project.yml").read_text(encoding="utf-8")
        match = re.search(r"CURRENT_PROJECT_VERSION:\s*(\d+)", xcodegen)

        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), expected)

        release_script = (ROOT / "scripts/release_macos.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('BUILD_NUMBER="${BUILD_NUMBER:-1}"', release_script)


if __name__ == "__main__":
    unittest.main()
