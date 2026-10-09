"""Optional dependency installation; every output stays under eval/."""
from __future__ import annotations

import io
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from .runtime import ROOT, configure, digest, write_json

SOURCES = {
    "bayeselo": ("https://www.remi-coulom.fr/Bayesian-Elo/bayeselo.tar.bz2", "bayeselo.tar.bz2",
                 "95f3d32381932ba9ccd1ab221ff33273cd3e48abedcfeba4c4f3f5a9b303a2a4"),
    "stockfish8": ("https://codeload.github.com/official-stockfish/Stockfish/zip/refs/tags/sf_8", "stockfish8.zip",
                   "a2a4e11f69394bd0faa7fd61612f5c2e954a57dba57d7a2d4f180cb2287a6b65"),
}


def setup() -> dict:
    configure()
    uv, compiler = shutil.which("uv"), shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("setup needs clang++ or g++ to build Stockfish 8 and BayesElo")
    if uv:
        command = [uv, "pip", "install", "--python", sys.executable, "--target", str(ROOT / "vendor/python"),
                   "-r", str(ROOT / "requirements.txt")]
    else:
        command = [sys.executable, "-B", "-m", "pip", "install", "--target", str(ROOT / "vendor/python"),
                   "-r", str(ROOT / "requirements.txt")]
    subprocess.run(command, cwd=ROOT, check=True)
    source, binaries = ROOT / "vendor/source", ROOT / "vendor/bin"
    source.mkdir(parents=True, exist_ok=True)
    binaries.mkdir(parents=True, exist_ok=True)
    manifest = {"platform": platform.platform(), "compiler": compiler, "sources": {}}
    for name, (url, filename, sha) in SOURCES.items():
        archive = ROOT / "vendor" / filename
        if not archive.exists():
            archive.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        if digest(archive) != sha:
            raise ValueError(f"source checksum mismatch: {archive}; remove it and rerun setup")
        data = archive.read_bytes()
        if filename.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for entry in z.infolist():
                    if not (source / entry.filename).resolve().is_relative_to(source.resolve()):
                        raise ValueError("unsafe archive path")
                z.extractall(source)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as t:
                t.extractall(source, filter="data")
        if name == "bayeselo":
            args = [compiler, "-std=c++98", "-O3", "-o", str(binaries / name), str(source / "BayesElo/bayeselo.cpp")]
        else:
            src = source / "Stockfish-sf_8/src"
            args = [compiler, "-std=c++11", "-O3", "-DNDEBUG", "-DIS_64BIT", "-DNO_PREFETCH", "-pthread",
                    "-o", str(binaries / name), *map(str, sorted(src.glob("*.cpp"))), str(src / "syzygy/tbprobe.cpp")]
        subprocess.run(args, cwd=ROOT, check=True)
        manifest["sources"][name] = {"url": url, "archive_sha256": sha,
                                      "binary_sha256": digest(binaries / name), "build_command": args}
    write_json("vendor/manifest.json", manifest)
    return manifest

