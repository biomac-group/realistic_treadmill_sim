"""Apply the Python-3.11 quoting fix when missing from the installed BioSym."""

from pathlib import Path

import biosym


settings_file = Path(biosym.__file__).parent / "ocp" / "utils" / "settings.py"
source = settings_file.read_text(encoding="utf-8")
patched = source.replace(
    '{settings["discretization"]["type"]}',
    "{settings['discretization']['type']}",
)
if patched != source:
    settings_file.write_text(patched, encoding="utf-8")
    print(f"Patched BioSym for Python 3.11: {settings_file}")
