"""Install hooks only in the independent Roop 3D repository."""
from pathlib import Path
import subprocess
from roop_3d.tools.git_guard import check_project

def install(root: Path) -> None:
    check_project(root)
    subprocess.run(['git','config','--local','core.hooksPath','.githooks'],cwd=root,check=True)

def main() -> int:
    try:
        install(Path(__file__).absolute().parents[3])
        print('Independent Roop 3D Git guards enabled',flush=True)
        return 0
    except Exception as error:
        print('ROOP3D_GUARD_ERROR: '+str(error),flush=True)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
