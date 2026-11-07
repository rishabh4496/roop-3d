"""Enforce the independent Roop 3D Git tree and dedicated remote identity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

class BoundaryError(RuntimeError):
    pass

GENERATED_PARTS = {'.venv','venv','env','__pycache__','node_modules','.runtime','.pytest_cache','.cache','.trt_cache','cache','outputs','dist','engines','trt_cache','.ruff_cache','.mypy_cache','logs'}
GENERATED_SUFFIXES = {'.pyc','.pyo','.engine','.trt','.plan','.part','.onnx','.pth','.pt','.safetensors','.log'}
GENERATED_FILES = {'protected_all_paths.json','stage00_verification.json','stage01_verification.json'}

def allowed_path(value: str) -> bool:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or chr(92) in value:
        return False
    if any(part.casefold() in GENERATED_PARTS for part in path.parts):
        return False
    if path.name.startswith('.env') and path.name != '.env.example':
        return False
    if path.name.casefold() in GENERATED_FILES:
        return False
    if path.suffix.casefold() in GENERATED_SUFFIXES:
        return False
    # In unified roop-3d, all repository source files in app/, roop_3d/, react-ui/, etc. are allowed
    return True

def git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(['git', *arguments], cwd=root)

def check_remote_url(url: str, expected: str = 'rishabh4496/roop-3d') -> None:
    match = re.fullmatch(r'(?:https://github[.]com/|git@github[.]com:)([^/]+/[^/]+?)/?', url)
    actual = match.group(1).removesuffix('.git') if match else ''
    if actual.casefold() != expected.casefold():
        raise BoundaryError('Push target is not the dedicated Roop 3D repository; Roop Ultimate is forbidden')

def check_project(root: Path) -> dict:
    marker = json.loads((root / '.roop3d-project.json').read_text('utf-8'))
    if marker.get('app_id') != 'roop-3d' or not (root / '.git').is_dir() or (root / '.git').is_symlink():
        raise BoundaryError('This must be the independent Roop 3D Git checkout')
    common = Path(git(root, 'rev-parse', '--git-common-dir').decode().strip())
    common = (common if common.is_absolute() else root / common).resolve()
    if common != (root / '.git').resolve() or (common / 'objects/info/alternates').exists():
        raise BoundaryError('Shared Git worktrees/object stores are forbidden')
    return marker

def check_names(names: list[str]) -> None:
    denied = [p for p in names if not allowed_path(p)]
    if denied:
        raise BoundaryError('Roop 3D boundary rejects: ' + ', '.join(denied))

def check_staged(root: Path) -> list[str]:
    check_project(root)
    names = [p.decode('utf-8') for p in git(root, 'diff', '--cached', '--name-only', '-z').split(bytes([0])) if p]
    check_names(names)
    for entry in git(root, 'ls-files', '--stage', '-z').split(bytes([0])):
        if entry:
            metadata, name = entry.split(bytes([9]), 1)
            if name.decode('utf-8') in names and metadata.split()[0] == b'120000':
                raise BoundaryError('Symlinks are forbidden in Roop 3D commits')
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True)
    return names

def check_branch(root: Path) -> str:
    marker = check_project(root)
    branch = git(root, 'branch', '--show-current').decode().strip()
    if not branch:
        raise BoundaryError('Detached HEAD is not a development branch')
    remote = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=root, capture_output=True, text=True)
    if remote.returncode == 0:
        check_remote_url(remote.stdout.strip(), marker['repository'])
    return branch

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--branch', action='store_true')
    parser.add_argument('--pre-push', action='store_true')
    parser.add_argument('remote_name', nargs='?')
    parser.add_argument('remote_url', nargs='?')
    args = parser.parse_args()
    root = Path(__file__).absolute().parents[3]
    try:
        marker = check_project(root)
        if args.branch or args.pre_push:
            check_branch(root)
        if args.staged:
            check_staged(root)
        if args.pre_push:
            if not args.remote_url:
                raise BoundaryError('The actual push remote URL is required')
            check_remote_url(args.remote_url, marker['repository'])
            names = [p.decode('utf-8') for p in git(root, 'ls-tree', '-r', '--name-only', '-z', 'HEAD').split(bytes([0])) if p]
            check_names(names)
            for line in sys.stdin:
                local_ref, local_sha, remote_ref, remote_sha = line.split()
                if not remote_ref.startswith('refs/heads/'):
                    raise BoundaryError('Only branch pushes are supported by this development guard')
        print('Independent Roop 3D Git boundary: PASS')
        return 0
    except (BoundaryError, OSError, ValueError, subprocess.SubprocessError) as error:
        print('Independent Roop 3D Git boundary: FAIL: ' + str(error), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
