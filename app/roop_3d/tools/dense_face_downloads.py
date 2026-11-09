"""Exact public upstream bytes used by the native dense-face export path.

Code and model/data licenses are separate; source references are recorded in
docs/native-dense-face-setup.md. Account-gated FLAME models are never fetched.
"""
from __future__ import annotations

from pathlib import Path
import urllib.request

from roop_3d.models.owned_files import owned_path

DDFA_REVISION = "e15385837dc1e051a6cf376b3827f2e279537b29"
HRN_REVISION = "58833698fd199fc08574a3cb9a8d8959fc58c213"
SMIRK_REVISION = "c7de404c4389f073906a6db1adabf62efcea3f35"

# family, relative path, public URL, SHA-256
PUBLIC_ASSETS = (
    ("3ddfa", "3ddfa/model/networks.py", f"https://raw.githubusercontent.com/wang-zidu/3DDFA-V3/{DDFA_REVISION}/model/networks.py", "21ba69b7eaf34154f9b83b0355ec7be5cab7fd1951ba24c1e67062c2c8729bf7"),
    ("3ddfa", "3ddfa/model/mb_v3_networks.py", f"https://raw.githubusercontent.com/wang-zidu/3DDFA-V3/{DDFA_REVISION}/model/mb_v3_networks.py", "aad74b63a2d2ed0e6fe63acbcd171ed37ef8faf49de0b73e6de3612b11c66329"),
    ("3ddfa", "face_model.npy", "https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/main/assets/face_model.npy", "f07ada9b4748ecdfeae3fa67aae88bb4b824ea7adc26853778c55d8dc0efa832"),
    ("3ddfa", "net_recon_mbnet.pth", "https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/main/assets/net_recon_mbnet.pth", "435f33e6fc93f3c0857538ffc5024d29bd6d09f98391d9be55f2d32b23cc2604"),
    ("3ddfa", "similarity_Lm3D_all.mat", "https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/main/assets/similarity_Lm3D_all.mat", "53b83ce6e35c50ddc3e97603650cef4970320c157e75c241c844f29c1dcba65a"),
    ("smirk", "smirk/src/smirk_encoder.py", f"https://raw.githubusercontent.com/georgeretsi/smirk/{SMIRK_REVISION}/src/smirk_encoder.py", "3806e867483d7d2d2fede7d34aed86f9fe49cff9827d90a9e72072dc94bf6c38"),
    ("smirk", "SMIRK_em1.pt", "https://drive.usercontent.google.com/download?id=1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE&export=download&confirm=t", "26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0"),
    ("smirk", "FLAME_masks.pkl", f"https://raw.githubusercontent.com/georgeretsi/smirk/{SMIRK_REVISION}/assets/FLAME_masks/FLAME_masks.pkl", "ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492"),
    ("smirk", "l_eyelid.npy", f"https://raw.githubusercontent.com/georgeretsi/smirk/{SMIRK_REVISION}/assets/l_eyelid.npy", "fa50997166c2f884fbfc577d99fe8707966763fc3d83a1870bb786df6cc9410d"),
    ("smirk", "r_eyelid.npy", f"https://raw.githubusercontent.com/georgeretsi/smirk/{SMIRK_REVISION}/assets/r_eyelid.npy", "dd78776fed765be4468889dd7d5b54b4bf61c8efbb2869b5b91409ce798aad8d"),
    ("hrn", "hrn/pix2pix/networks.py", f"https://raw.githubusercontent.com/youngLBW/HRN/{HRN_REVISION}/pix2pix/networks.py", "62cd32ba7d43df0c02207ece37062ca78c9e61c15ace763d774281a22a745dde"),
    ("hrn", "epoch_10.pth", "https://drive.usercontent.google.com/download?id=1MM-hHVqfK-ufESOWVUDgS6m7Ezlerf8E&export=download&confirm=t", "bb1e59c3883810113561c6da412b7d5c8627b1481051ca176db2e8c0870fa73e"),
    ("hrn", "bfm_uvs2.npy", "https://drive.usercontent.google.com/download?id=1ubHwchKq_FwqrW04j7ezDRQ0-BlqvWce&export=download&confirm=t", "7024ccd4975f9909c63c2c0772399e55d9c1049828bcea48172d581f68418b26"),
)


def fetch_public(args):
    from roop_3d.tools.prepare_dense_face import sha256
    output = owned_path(Path(args.output))
    for family, relative, url, expected in PUBLIC_ASSETS:
        if args.component not in ("all", family):
            continue
        destination = owned_path(output / relative)
        if destination.is_file():
            if sha256(destination) != expected:
                raise ValueError(f"Existing upstream asset has different bytes: {destination}")
            print(f"Verified {relative}", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = owned_path(destination.with_name(destination.name+".part"))
        if partial.exists():
            raise ValueError(f"An unfinished download exists: {partial}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError(f"Upstream returned HTML instead of {relative}")
                with partial.open("xb") as stream:
                    total = 0
                    while chunk := response.read(1048576):
                        total += len(chunk)
                        if total > 512 * 1048576:
                            raise ValueError("Upstream asset exceeds the known checkpoint size budget")
                        stream.write(chunk)
            if sha256(partial) != expected:
                raise ValueError(f"Upstream digest changed: {relative}")
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        print(f"Downloaded and verified {relative}", flush=True)
