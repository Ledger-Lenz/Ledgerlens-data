"""CI entry point: fail if any registered importer is non-conformant."""

import sys

from ingestion import registered_importers  # noqa: F401  (registers built-ins)
from ingestion.importer_registry import get_registry

registry = get_registry()
failures = registry.verify_conformance()
for f in failures:
    print("FAIL:", f)
print(f"{len(registry.list_all())} importer(s) checked, {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
