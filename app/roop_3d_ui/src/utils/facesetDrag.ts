import type { DragEvent } from 'react';

/** Drag payload for "drag a FaceSet onto a target track". A private MIME type, so random text/file drops never assign. */
export const FACESET_MIME = 'application/x-roop3d-faceset';
const ID = /^[A-Za-z0-9_-]{1,64}$/;

export function startFacesetDrag(event: DragEvent, facesetId: string, name: string) {
  event.dataTransfer.setData(FACESET_MIME, facesetId);
  event.dataTransfer.setData('text/plain', name);
  event.dataTransfer.effectAllowed = 'copy';
}

/** True while a FaceSet (not a file or arbitrary text) is being dragged over the element. */
export const carriesFaceset = (event: DragEvent) => Array.from(event.dataTransfer?.types ?? []).includes(FACESET_MIME);

/** The dropped FaceSet id, or null for anything that is not a well-formed FaceSet drag. */
export function droppedFaceset(event: DragEvent): string | null {
  const value = event.dataTransfer?.getData(FACESET_MIME) ?? '';
  return ID.test(value) ? value : null;
}

/**
 * Props that turn any element into a FaceSet drop target. `onDropFaceset` receives a validated id; `setOver` lets the
 * element show a highlight only while a FaceSet hovers it.
 */
export function facesetDropProps(onDropFaceset: (facesetId: string) => void, setOver?: (over: boolean) => void) {
  return {
    onDragOver: (event: DragEvent) => { if (!carriesFaceset(event)) return; event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; setOver?.(true); },
    onDragLeave: () => setOver?.(false),
    onDrop: (event: DragEvent) => {
      setOver?.(false);
      const id = droppedFaceset(event);
      if (!id) return;
      event.preventDefault();
      event.stopPropagation();
      onDropFaceset(id);
    },
  };
}
