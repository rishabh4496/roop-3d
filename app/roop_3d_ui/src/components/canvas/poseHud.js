import { formatPose, toStandardPose } from '../../utils/poseCells';

/** Standard-frame pose of a detected face (prefers the server's `pose3d`; otherwise converts the legacy `pose`). */
export function facePose(face) {
  return face.pose3d ? toStandardPose(face.pose3d, 'standard') : toStandardPose(face.pose, 'legacy');
}

// Eyes, nose, mouth corners: coloured by anatomy so a flipped or misordered landmark set is obvious at a glance.
export const LANDMARK_COLORS = ['#38bdf8', '#38bdf8', '#facc15', '#f472b6', '#f472b6'];

/**
 * Draws, for EVERY detected face in the frame, its 5-point ArcFace landmarks and a live yaw / pitch / roll badge.
 * Boxes are drawn by the caller; this layer only adds what the HUD toggles control.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array<any>} faces
 * @param {{ mapPoint: (p: number[]) => {x:number,y:number}, mapBox: (b: number[]) => {x:number,y:number,w:number,h:number},
 *   bounds: { offsetX: number, offsetY: number, width: number, height: number },
 *   landmarks?: boolean, pose?: boolean }} options
 */
export function drawPoseHud(ctx, faces, { mapPoint, mapBox, bounds, landmarks = true, pose = true }) {
  const drawn = [];
  for (const face of faces) {
    if (landmarks && Array.isArray(face.landmarks5) && face.landmarks5.length === 5) {
      ctx.save();
      face.landmarks5.forEach((point, index) => {
        const { x, y } = mapPoint(point);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = LANDMARK_COLORS[index];
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(0,0,0,0.65)';
        ctx.stroke();
      });
      ctx.restore();
      drawn.push({ id: face.id, kind: 'landmarks' });
    }
    const attitude = pose ? facePose(face) : null;
    if (attitude) {
      const box = mapBox(face.bbox);
      const text = formatPose(attitude);
      ctx.save();
      ctx.font = '600 10px ui-monospace, Menlo, Consolas, monospace';
      const width = ctx.measureText(text).width + 10, height = 16;
      const left = Math.max(bounds.offsetX, Math.min(box.x + box.w - width, bounds.offsetX + bounds.width - width));
      const top = box.y - height - 2 >= bounds.offsetY ? box.y - height - 2 : box.y + box.h + 2;
      ctx.fillStyle = 'rgba(10,16,18,0.88)';
      ctx.fillRect(left, top, width, height);
      ctx.fillStyle = '#e8f1f2';
      ctx.fillText(text, left + 5, top + 11);
      ctx.restore();
      drawn.push({ id: face.id, kind: 'pose', text });
    }
  }
  return drawn;
}
