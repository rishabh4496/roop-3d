import { useState, useCallback, type DragEvent } from 'react';
import type { Probe } from '../../contracts';
import { fileName } from '../../utils/safeData';
import * as apiModule from '../../api';
import { useTargetMedia } from '../../stores/targetMedia';
import './targetMedia.css';

const getMediaURL = (path: string): string => {
  try {
    if (typeof (apiModule as any)?.mediaURL === 'function') {
      return (apiModule as any).mediaURL(path);
    }
  } catch {
    // fallback
  }
  return `/api/v1/media/content?path=${encodeURIComponent(path)}`;
};

export interface TargetMediaCardProps {
  target: string;
  probe: Probe | null;
  busy: boolean;
  uploadProgress?: number;
  onAddTarget: () => void;
  onReplaceTarget: () => void;
  onRemoveTarget: () => void;
  onDropFiles: (files: File[]) => void;
}

function formatClock(seconds: number) {
  const safe = Math.max(0, seconds || 0);
  return `${Math.floor(safe / 60).toString().padStart(2, '0')}:${(safe % 60).toFixed(2).padStart(5, '0')}`;
}

export default function TargetMediaCard({
  target,
  probe,
  busy,
  uploadProgress = 0,
  onAddTarget,
  onReplaceTarget,
  onRemoveTarget,
  onDropFiles,
}: TargetMediaCardProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const targetInfo = useTargetMedia(state => state);

  const isImage = target ? /\.(png|jpe?g|webp|bmp)$/i.test(target) : targetInfo.mediaType === 'image';
  const name = target ? fileName(target) : targetInfo.filename || '';
  const previewSrc = targetInfo.objectUrl || (target ? getMediaURL(target) : '');

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) {
      onDropFiles(files);
    }
  }, [onDropFiles]);

  if (!target) {
    return (
      <div className="target-media-container" data-testid="target-media-empty">
        <div
          className={`target-dropzone ${isDragOver ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragEnter={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={onAddTarget}
          role="button"
          tabIndex={0}
          aria-label="Target media dropzone"
          data-testid="target-dropzone"
        >
          <span className="target-dropzone-icon" aria-hidden="true">⇪</span>
          <div className="target-dropzone-text">
            <strong>Drag & drop target media here</strong>
            <small className="muted">or click to browse from disk</small>
          </div>
          <button
            type="button"
            className="button small primary-soft"
            disabled={busy}
            onClick={e => {
              e.stopPropagation();
              onAddTarget();
            }}
            data-testid="add-target-media-button"
          >
            ＋ Add Target Media
          </button>
          <div className="target-dropzone-badges">
            <span className="target-dropzone-badge">VIDEO (MP4, MOV, WEBM)</span>
            <span className="target-dropzone-badge">IMAGE (PNG, JPG, WEBP)</span>
          </div>
        </div>
        {uploadProgress > 0 && uploadProgress < 1 && (
          <div className="target-upload-progress" role="progressbar" aria-valuenow={Math.round(uploadProgress * 100)}>
            <div className="target-upload-bar" style={{ width: `${Math.round(uploadProgress * 100)}%` }} />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="target-media-container" data-testid="target-media-active">
      <div className="target-preview-card">
        <div className="target-preview-header">
          <span className={`target-type-badge ${isImage ? 'image' : 'video'}`}>
            {isImage ? 'IMAGE' : 'VIDEO'}
          </span>
          <span className="target-filename" title={name} data-testid="target-filename">
            {name}
          </span>
          <div className="target-preview-actions">
            <button
              type="button"
              className="target-action-btn"
              onClick={onReplaceTarget}
              title="Replace Target Media"
              disabled={busy}
              data-testid="replace-target-button"
            >
              🔁 Replace
            </button>
            <button
              type="button"
              className="target-action-btn danger"
              onClick={onRemoveTarget}
              title="Remove Target (Clear)"
              disabled={busy}
              data-testid="remove-target-button"
            >
              🗑️ Remove
            </button>
          </div>
        </div>

        {previewSrc && (
          <div className="target-preview-thumb-box">
            {isImage ? (
              <img
                src={previewSrc}
                alt={name}
                className="target-preview-thumb"
                data-testid="target-image-thumbnail"
              />
            ) : (
              <video
                src={previewSrc}
                className="target-preview-thumb"
                muted
                preload="metadata"
                data-testid="target-video-thumbnail"
              />
            )}
            {probe && probe.width && probe.height ? (
              <span className="target-preview-res-badge">
                {probe.width} × {probe.height}
              </span>
            ) : null}
          </div>
        )}

        <div className="target-preview-details">
          <span>
            {probe ? `${probe.width} × ${probe.height}` : 'Resolution unknown'}
          </span>
          <span>
            {isImage
              ? 'Single Frame'
              : probe
              ? `${formatClock(probe.duration_seconds ?? 0)} · ${probe.fps_rational ?? '?'} fps`
              : ''}
          </span>
        </div>

        {uploadProgress > 0 && uploadProgress < 1 && (
          <div className="target-upload-progress" role="progressbar" aria-valuenow={Math.round(uploadProgress * 100)}>
            <div className="target-upload-bar" style={{ width: `${Math.round(uploadProgress * 100)}%` }} />
          </div>
        )}
      </div>
    </div>
  );
}
