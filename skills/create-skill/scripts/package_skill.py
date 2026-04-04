#!/usr/bin/env python3
"""Package a skill directory into a distributable .skill file.

Usage:
    python scripts/package_skill.py <skill-directory> [output-directory]
    python scripts/package_skill.py --help

The .skill file is a ZIP archive containing the skill directory.
Excludes: __pycache__, *.pyc, .DS_Store, evals/ (test data).
"""

import fnmatch
import sys
import zipfile
from pathlib import Path

# Import our validator
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from validate_skill import validate_skill

# Exclusion patterns
EXCLUDE_DIRS = {"__pycache__", "node_modules", ".git"}
EXCLUDE_GLOBS = {"*.pyc", "*.pyo"}
EXCLUDE_FILES = {".DS_Store", "Thumbs.db"}
ROOT_EXCLUDE_DIRS = {"evals"}  # Only at skill root level


def should_exclude(rel_path: Path) -> bool:
    """Check if a path should be excluded from packaging."""
    parts = rel_path.parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return True
    # rel_path is relative to skill parent, so parts[0] is skill dir, parts[1] is subdir
    if len(parts) > 1 and parts[1] in ROOT_EXCLUDE_DIRS:
        return True
    name = rel_path.name
    if name in EXCLUDE_FILES:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_GLOBS)


def package_skill(skill_path: str, output_dir: str = None) -> Path:
    """
    Package a skill folder into a .skill file.

    Args:
        skill_path: Path to the skill folder
        output_dir: Optional output directory (defaults to cwd)

    Returns:
        Path to the created .skill file

    Raises:
        ValueError: If validation fails
    """
    skill_path = Path(skill_path).resolve()
    skill_name = skill_path.name

    # Validate first
    result = validate_skill(str(skill_path))
    if not result.passed:
        raise ValueError(f"Validation failed:\n{result.summary()}")

    print(f"✅ Validation passed")

    # Determine output path
    if output_dir:
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = Path.cwd()

    skill_file = out / f"{skill_name}.skill"

    # Create ZIP
    file_count = 0
    with zipfile.ZipFile(skill_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(skill_path.rglob('*')):
            if not file_path.is_file():
                continue
            arcname = file_path.relative_to(skill_path.parent)
            if should_exclude(arcname):
                print(f"  Skipped: {arcname}")
                continue
            zf.write(file_path, arcname)
            print(f"  Added:   {arcname}")
            file_count += 1

    size_kb = skill_file.stat().st_size / 1024
    print(f"\n📦 Packaged {file_count} files → {skill_file} ({size_kb:.1f} KB)")
    return skill_file


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    skill_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = package_skill(skill_path, output_dir)
        print(f"\n✅ Done: {result}")
    except ValueError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
