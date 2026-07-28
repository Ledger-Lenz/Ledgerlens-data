"""Tests for the offline dependency lockfile validation contract."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.validate_lockfiles import stamp_pair, validate_pair, validate_python_policy


class LockfileValidationTests(unittest.TestCase):
    HASH = "a" * 64

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.manifest = self.root / "requirements.in"
        self.lockfile = self.root / "requirements.txt"

    def _write_pair(
        self,
        manifest: str = "Demo_Package>=1.0\nrequests>=2\n",
        requirements: str | None = None,
    ) -> None:
        if requirements is None:
            requirements = (
                f"demo-package==1.2.3 --hash=sha256:{self.HASH}\n"
                f"requests==2.32.0 --hash=sha256:{self.HASH}\n"
                f"urllib3==2.2.2 --hash=sha256:{self.HASH}\n"
            )
        self.manifest.write_text(manifest, encoding="utf-8")
        digest = hashlib.sha256(manifest.encode()).hexdigest()
        self.lockfile.write_text(
            "# lockfile-version: 1\n"
            "# source: requirements.in\n"
            f"# source-sha256: {digest}\n"
            "# resolver: uv==0.8.3\n"
            "# target: universal-python>=3.11,<3.13\n"
            f"{requirements}",
            encoding="utf-8",
        )

    def test_valid_lock_accepts_normalized_names_and_transitive_dependencies(self) -> None:
        self._write_pair()

        self.assertEqual(validate_pair(self.manifest, self.lockfile), [])

    def test_stale_source_fingerprint_is_actionable(self) -> None:
        self._write_pair()
        self.manifest.write_text("demo-package>=2\nrequests>=2\n", encoding="utf-8")

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(any("stale source fingerprint" in error for error in errors))
        self.assertTrue(any("--stamp" in error for error in errors))

    def test_missing_direct_dependency_is_rejected(self) -> None:
        self._write_pair(requirements="demo-package==1.2.3\n")

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(any("missing direct dependency 'requests'" in error for error in errors))

    def test_range_in_lockfile_is_rejected(self) -> None:
        self._write_pair(
            requirements=(
                f"demo-package>=1.2.3 --hash=sha256:{self.HASH}\n"
                f"requests==2.32.0 --hash=sha256:{self.HASH}\n"
            )
        )

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(any("'demo-package' is not locked" in error for error in errors))

    def test_duplicate_normalized_name_is_rejected_with_lines(self) -> None:
        self._write_pair(
            requirements=(
                f"demo-package==1.2.3 --hash=sha256:{self.HASH}\n"
                f"requests==2.32.0 --hash=sha256:{self.HASH}\n"
                f"Requests==2.31.0 --hash=sha256:{self.HASH}\n"
            )
        )

        errors = validate_pair(self.manifest, self.lockfile)

        duplicate = next(error for error in errors if "duplicate requirement 'requests'" in error)
        self.assertIn(":8:", duplicate)
        self.assertIn("line 7", duplicate)

    def test_missing_lockfile_is_rejected(self) -> None:
        self.manifest.write_text("demo-package>=1\n", encoding="utf-8")

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertEqual(len(errors), 1)
        self.assertIn("lockfile does not exist", errors[0])

    def test_invalid_or_duplicate_headers_are_rejected(self) -> None:
        self._write_pair()
        with self.lockfile.open("a", encoding="utf-8") as handle:
            handle.write("# lockfile-version: 1\n")

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(
            any("exactly one valid lockfile-version header" in error for error in errors)
        )

    def test_lock_pin_must_satisfy_manifest_specifier(self) -> None:
        self._write_pair(
            manifest="requests>=2.31\n",
            requirements=f"requests==2.0 --hash=sha256:{self.HASH}\n",
        )

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(
            any("does not satisfy manifest specifier '>=2.31'" in error for error in errors)
        )

    def test_lock_pin_satisfies_bounded_manifest_specifier(self) -> None:
        self._write_pair(
            manifest="requests>=2.31,<3\n",
            requirements=f"requests==2.32.0 --hash=sha256:{self.HASH}\n",
        )

        self.assertEqual(validate_pair(self.manifest, self.lockfile), [])

    def test_unhashed_lock_entry_is_rejected(self) -> None:
        self._write_pair(requirements="demo-package==1.2.3\nrequests==2.32.0\n")

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(
            any("'demo-package' has no sha256 artifact hash" in error for error in errors)
        )
        self.assertTrue(any("'requests' has no sha256 artifact hash" in error for error in errors))

    def test_malformed_hash_is_rejected(self) -> None:
        self._write_pair(
            requirements=(
                "demo-package==1.2.3 --hash=sha256:not-a-digest\n"
                f"requests==2.32.0 --hash=sha256:{self.HASH}\n"
            )
        )

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(any("malformed or unsupported hash option" in error for error in errors))

    def test_direct_extra_set_mismatch_is_rejected(self) -> None:
        self._write_pair(
            manifest="demo-package[security]>=1\n",
            requirements=(
                f"demo-package[crypto, security]==1.2.3 --hash=sha256:{self.HASH}\n"
                f"crypto-helper==4.0 --hash=sha256:{self.HASH}\n"
            ),
        )

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(
            any("missing direct dependency 'demo-package'" in error for error in errors)
        )

    def test_direct_extras_are_order_and_whitespace_insensitive(self) -> None:
        self._write_pair(
            manifest="demo-package[security,crypto]>=1\n",
            requirements=(
                f"demo-package[crypto, security]==1.2.3 --hash=sha256:{self.HASH}\n"
                f"crypto-helper==4.0 --hash=sha256:{self.HASH}\n"
            ),
        )

        self.assertEqual(validate_pair(self.manifest, self.lockfile), [])

    def test_universal_marker_variants_must_all_satisfy_policy(self) -> None:
        self._write_pair(
            manifest="demo-package>=1\n",
            requirements=(
                f'demo-package==1.2.3 ; python_version < "3.13" '
                f"--hash=sha256:{self.HASH}\n"
                f'demo-package==0.9 ; python_version >= "3.13" '
                f"--hash=sha256:{self.HASH}\n"
            ),
        )

        errors = validate_pair(self.manifest, self.lockfile)

        self.assertTrue(
            any("does not satisfy manifest specifier '>=1'" in error for error in errors)
        )

    def test_stamp_refreshes_only_the_source_fingerprint(self) -> None:
        self._write_pair()
        self.manifest.write_text("demo-package>=1.1\nrequests>=2\n", encoding="utf-8")
        before = self.lockfile.read_text(encoding="utf-8")

        self.assertIsNone(stamp_pair(self.manifest, self.lockfile))

        after = self.lockfile.read_text(encoding="utf-8")
        expected = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.assertIn(f"# source-sha256: {expected}", after)
        self.assertEqual(
            before.replace(before.splitlines()[2], ""),
            after.replace(after.splitlines()[2], ""),
        )

    def test_stamp_adds_headers_to_a_newly_generated_lock(self) -> None:
        self.manifest.write_text("demo-package>=1\n", encoding="utf-8")
        self.lockfile.write_text(
            "# This file is autogenerated by pip-compile\n"
            "demo-package==1.2.3 \\\n"
            "    --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
            encoding="utf-8",
        )

        self.assertIsNone(stamp_pair(self.manifest, self.lockfile))

        self.assertEqual(validate_pair(self.manifest, self.lockfile), [])

    def test_stamp_records_resolver_and_universal_target(self) -> None:
        self.manifest.write_text("demo-package>=1\n", encoding="utf-8")
        self.lockfile.write_text(
            f"demo-package==1.2.3 --hash=sha256:{self.HASH}\n",
            encoding="utf-8",
        )

        self.assertIsNone(stamp_pair(self.manifest, self.lockfile))

        stamped = self.lockfile.read_text(encoding="utf-8")
        self.assertIn("# resolver: uv==0.8.3", stamped)
        self.assertIn("# target: universal-python>=3.11,<3.13", stamped)

    def test_supported_python_policy_is_accepted(self) -> None:
        pyproject = self.root / "pyproject.toml"
        pyproject.write_text('[project]\nrequires-python = ">=3.11,<3.13"\n', encoding="utf-8")

        self.assertEqual(validate_python_policy(pyproject), [])

    def test_python_policy_drift_is_rejected(self) -> None:
        pyproject = self.root / "pyproject.toml"
        pyproject.write_text('[project]\nrequires-python = ">=3.11"\n', encoding="utf-8")

        errors = validate_python_policy(pyproject)

        self.assertTrue(any("configured Python 3.11/3.12 locks" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
