import os
import re
from pathlib import Path


WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def resolve_project_dir(anchor_file: str, levels_up: int = 0) -> str:
    fallback = Path(anchor_file).resolve().parent
    for _ in range(levels_up):
        fallback = fallback.parent

    configured_path = os.getenv("PROJECT_DIR", "").strip()
    if configured_path and _is_usable_project_dir(configured_path):
        return str(Path(configured_path).expanduser().resolve())

    return str(fallback)


def _is_usable_project_dir(raw_path: str) -> bool:
    if os.name != "nt" and WINDOWS_DRIVE_PATH_RE.match(raw_path):
        return False

    path = Path(raw_path).expanduser()
    return path.is_absolute() and path.is_dir()
