"""Prepare a real-photo ArcFace relighting fixture from an owned dense FaceSet.

Run from app/: python -m roop_3d.tools.prepare_relighting_case facesets/source.fsz
This measures identity retention through relighting of a real photograph; it
does not measure the identity accuracy of a separate neural identity swap.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.models.owned_files import owned_path
from roop_3d.reconstruction.full_head import GPUHeadRasterizer, HeadPose, head_roi, solve_dense_pose
from roop_3d.reconstruction.makehuman_template import makehuman_template
from roop_3d.reconstruction.photometry import fit_target_surface


def prepare(faceset, output):
    path = owned_path(Path(faceset).absolute())
    output = owned_path(Path(output).absolute())
    document = FaceSetReader().read(path)
    views = [view for view in document.views if view.landmarks_68 is not None]
    if not views:
        raise ValueError('Real relighting fixture requires measured 68-point landmarks in the FaceSet')
    view = min(views, key=lambda v: abs(v.yaw)+abs(v.pitch))
    image = cv2.imdecode(np.frombuffer(view.image_bytes,np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Source FaceSet image could not be decoded')
    h,w = image.shape[:2]
    k = np.array([[max(h,w)*1.5,0,w/2],[0,max(h,w)*1.5,h/2],[0,0,1]],np.float64)
    mesh = makehuman_template()
    ids = mesh.landmark_vertices['68pt']
    points = np.asarray(view.landmarks_68,np.float64)[:,:2]
    rv,tv = solve_dense_pose(mesh.vertices[ids[17:48]]*(1,1,-1),points[17:48],k)
    pose = HeadPose(cv2.Rodrigues(rv)[0],tv.ravel(),k)
    vertices = fit_target_surface(mesh.vertices,ids,points,pose)
    inverse = head_roi(pose.project(vertices),512)
    gpu = GPUHeadRasterizer()
    try:
        buffers = gpu.render_buffers(vertices,mesh.faces,mesh.uvs,np.zeros((64,64,3),np.uint8),
            np.ones((64,64),np.float32),pose,inverse,512)
    finally:
        gpu.release()
    frame_to_crop = cv2.invertAffineTransform(inverse)
    crop = cv2.warpAffine(image,frame_to_crop,(512,512))
    points5 = cv2.transform(np.asarray(view.landmarks,np.float32)[None],frame_to_crop)[0]
    output.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(owned_path(output/'identity-roi.png')),crop):
        raise OSError('Could not write identity ROI')
    np.savez_compressed(owned_path(output/'fitted-makehuman-buffers.npz'),normals=buffers.normals,
                        depth=buffers.depth,inverse=inverse,camera=k,
                        landmarks68=cv2.transform(points[None].astype(np.float32),frame_to_crop)[0])
    manifest = owned_path(output/'case.json')
    manifest.write_text(json.dumps({'image':'identity-roi.png','buffers':'fitted-makehuman-buffers.npz',
        'landmarks5':points5.tolist(),'source_sha256':document.source_sha256,
        'scope':'real_photo_identity_retention_through_relighting'},indent=2),encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('faceset',type=Path)
    parser.add_argument('--output',type=Path,default=application_root()/'roop_3d/.runtime/acceptance/photometric-head')
    args = parser.parse_args()
    print(prepare(args.faceset,args.output))


if __name__ == '__main__':
    main()
