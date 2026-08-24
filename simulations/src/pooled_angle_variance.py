"""Discover the speed series belonging to a selected angle reference."""

from __future__ import annotations

import re
from pathlib import Path


SPEED_TOKEN = re.compile(r"speed_\d+p\d+")


def add_pooled_variance_files(collocation_settings, reference_file):
    reference = Path(reference_file)
    pattern = SPEED_TOKEN.sub("speed_*", reference.name)
    files = [str(path) for path in sorted(reference.parent.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No angle speed series matches {reference}")
    for objective in collocation_settings.get("objectives", []):
        if objective.get("name") == "track_angles":
            objective.setdefault("args", {})["pooled_variance_files"] = files
    print(f"Using angle variance pooled across {len(files)} speed references: {files}")
    return files
