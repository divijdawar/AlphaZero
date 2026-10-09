from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def configure() -> None:
    sys.dont_write_bytecode = True
    cache, tmp = ROOT / "work/cache", ROOT / "work/tmp"
    cache.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    for key, value in {
        "PYTHONDONTWRITEBYTECODE": "1", "XDG_CACHE_HOME": str(cache),
        "CACHEDB": str(cache / "tinygrad/cache.db"), "TMPDIR": str(tmp),
        "CLANG_MODULE_CACHE_PATH": str(cache / "clang"), "CUDA_CACHE_PATH": str(cache / "cuda"),
        "UV_CACHE_DIR": str(cache / "uv"),
    }.items():
        os.environ[key] = value
    tempfile.tempdir = str(tmp)
    sys.path.insert(0, str(ROOT / "vendor/python"))


def output_path(path: str | Path) -> Path:
    path = Path(path)
    result = (path if path.is_absolute() else ROOT / path).resolve()
    if result == ROOT or not result.is_relative_to(ROOT):
        raise ValueError(f"outputs must stay inside {ROOT}: {path}")
    return result


def write_json(path: str | Path, value: object) -> None:
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        temporary = Path(f.name)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def outside_snapshot() -> dict[str, str]:
    result = {}
    for directory, names, files in os.walk(REPO):
        names[:] = [n for n in names if n != ".git" and (Path(directory) / n).resolve() != ROOT
                    and not (Path(directory) / n).is_symlink()]
        for name in files:
            p = Path(directory) / name
            if p.is_file() and not p.is_symlink():
                result[str(p.relative_to(REPO))] = digest(p)
    return result


def isolation_audit() -> dict:
    before = json.loads((ROOT / "work/isolation-before.json").read_text())
    after = outside_snapshot()
    changes = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    return {"outside_files_checked": len(after), "outside_changes": changes, "ok": not changes}

