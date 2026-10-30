from pathlib import Path
import json
import subprocess
import pytest
from roop_3d.tools.git_guard import BoundaryError, allowed_path, check_staged, check_branch, check_remote_url
from roop_3d.tools.install_git_guard import install

def make_repo(root: Path) -> None:
    subprocess.run(['git','init','--quiet',str(root)],check=True)
    (root/'.roop3d-project.json').write_text(json.dumps({'app_id':'roop-3d','repository':'rishabh4496/roop-3d'}),encoding='utf-8')

def test_boundary_accepts_only_independent_app_source() -> None:
    for path in ('app/roop_3d/server/app.py','app/roop_3d_ui/src/main.tsx','start.js','pinokio.js','.gitignore'):
        assert allowed_path(path)
    for path in ('app/legacy.py','face_engine/models.py','react-ui/src/App.jsx','../roop-ultimate/app/settings.py','app/roop_3d/.venv/x','app/roop_3d/.env','app/roop_3d/weights.onnx','app/roop_3d/x.engine','app/roop_3d/../legacy.py','app/roop_3d_ui/node_modules/x','app/roop_3d/audit/protected_all_paths.json'):
        assert not allowed_path(path),path

def test_guard_reads_real_git_index(tmp_path: Path) -> None:
    make_repo(tmp_path)
    (tmp_path/'app/roop_3d').mkdir(parents=True)
    (tmp_path/'app/roop_3d/service.py').write_bytes(b'source')
    subprocess.run(['git','add','app/roop_3d/service.py'],cwd=tmp_path,check=True)
    assert check_staged(tmp_path)==['app/roop_3d/service.py']
    (tmp_path/'app/legacy.py').write_bytes(b'forbidden')
    subprocess.run(['git','add','app/legacy.py'],cwd=tmp_path,check=True)
    with pytest.raises(BoundaryError,match='legacy.py'):
        check_staged(tmp_path)

def test_dedicated_remote_is_required() -> None:
    for url in ('https://github.com/rishabh4496/roop-3d.git','git@github.com:rishabh4496/roop-3d.git'):
        check_remote_url(url)
    for url in ('https://github.com/rishabh4496/roop-ultimate.git','git@github.com:rishabh4496/roop-ultimate.git','../roop-ultimate','https://example.com/rishabh4496/roop-3d'):
        with pytest.raises(BoundaryError):
            check_remote_url(url)

def test_hooks_modify_only_this_repository_and_reject_wrong_origin(tmp_path: Path) -> None:
    make_repo(tmp_path)
    install(tmp_path)
    assert subprocess.check_output(['git','config','--get','core.hooksPath'],cwd=tmp_path,text=True).strip()=='.githooks'
    subprocess.run(['git','remote','add','origin','https://github.com/rishabh4496/roop-ultimate.git'],cwd=tmp_path,check=True)
    with pytest.raises(BoundaryError,match='forbidden'):
        check_branch(tmp_path)
