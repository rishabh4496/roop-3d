import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api, post } from '../../api';
import type { FaceSet } from '../../contracts';
import VirtualList from '../common/VirtualList';

export interface FacesetLibraryProps {
  sources?: FaceSet[];
  selectedSourceId?: string;
  onSelectSource?: (facesetId: string) => void;
  onRefreshSources?: () => Promise<void>;
  onImportFaceset?: (file: File) => void;
  canSave?: boolean;
  onSaveFace?: (name: string) => Promise<void>;
  notify?: (msg: string, type?: 'info' | 'error') => void;
}

export default function FacesetLibrary({
  sources = [],
  selectedSourceId,
  onSelectSource,
  onRefreshSources,
  onImportFaceset,
  canSave = false,
  onSaveFace,
  notify,
}: FacesetLibraryProps) {
  const [open, setOpen] = useState(true);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState('');
  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState('');
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [newFacesetName, setNewFacesetName] = useState('');

  const importRef = useRef<HTMLInputElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);
  const renameDone = useRef(false);

  const selected = sources.find((s) => s.faceset_id === selectedSourceId) || null;

  const q = filter.trim().toLowerCase();
  const shown = q ? sources.filter((e) => e.name.toLowerCase().includes(q)) : sources;

  const refresh = useCallback(async () => {
    if (onRefreshSources) {
      setBusy(true);
      try {
        await onRefreshSources();
      } catch (err: any) {
        notify?.(err.message, 'error');
      } finally {
        setBusy(false);
      }
    }
  }, [onRefreshSources, notify]);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!pickerOpen) return;
    const onDown = (ev: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(ev.target as Node)) {
        setPickerOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [pickerOpen]);

  const handleSaveCurrent = async () => {
    if (!newFacesetName.trim()) return;
    setBusy(true);
    try {
      if (onSaveFace) {
        await onSaveFace(newFacesetName.trim());
      }
      setSaveModalOpen(false);
      setNewFacesetName('');
      await refresh();
      notify?.(`Saved “${newFacesetName.trim()}” to FaceSet library`);
    } catch (err: any) {
      notify?.(err.message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const load = (entry: FaceSet) => {
    onSelectSource?.(entry.faceset_id);
    setPickerOpen(false);
    notify?.(`Loaded “${entry.name}” into source identities`);
  };

  const del = async (entry: FaceSet) => {
    if (!window.confirm(`Delete FaceSet “${entry.name}”? This removes the file on disk.`)) return;
    setBusy(true);
    try {
      await api(`/facesets/${entry.faceset_id}`, { method: 'DELETE' });
      await refresh();
      notify?.(`Deleted “${entry.name}”`);
    } catch (err: any) {
      notify?.(err.message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const beginRename = (entry: FaceSet) => {
    renameDone.current = false;
    setRenaming(entry.faceset_id);
    setRenameVal(entry.name);
  };

  const cancelRename = () => {
    renameDone.current = true;
    setRenaming(null);
  };

  const commitRename = async (entry: FaceSet) => {
    if (renameDone.current) return;
    renameDone.current = true;
    const nextName = renameVal.trim();
    if (!nextName || nextName === entry.name) {
      setRenaming(null);
      return;
    }
    setBusy(true);
    try {
      await post(`/facesets/${entry.faceset_id}/rename`, { name: nextName });
      await refresh();
      notify?.(`Renamed to “${nextName}”`);
    } catch (err: any) {
      notify?.(err.message, 'error');
    } finally {
      setRenaming(null);
      setBusy(false);
    }
  };

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    e.target.value = '';
    if (!files.length) return;
    for (const f of files) {
      onImportFaceset?.(f);
    }
  };

  const openFolder = async () => {
    try {
      await post('/facesets/open-folder', {});
      notify?.('Opened FaceSets library folder in explorer');
    } catch (err: any) {
      notify?.(err.message, 'error');
    }
  };

  const rebuildThumbs = async () => {
    setBusy(true);
    try {
      const res = await post<{ rebuilt: number }>('/facesets/rebuild-thumbs', {});
      await refresh();
      notify?.(`Rebuilt thumbnails for ${res.rebuilt} FaceSet(s)`);
    } catch (err: any) {
      notify?.(err.message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const Thumb = ({ entry, size = 'w-7 h-7' }: { entry: FaceSet | null; size?: string }) => {
    const [imgErr, setImgErr] = useState(false);
    return (
      <span className={`shrink-0 ${size} rounded-md overflow-hidden bg-black/40 border border-white/10 inline-flex items-center justify-center`}>
        {entry && !imgErr ? (
          <img
            src={`/api/v1/facesets/${entry.faceset_id}/thumb`}
            alt={entry.name}
            onError={() => setImgErr(true)}
            className="w-full h-full object-cover"
            draggable={false}
          />
        ) : (
          <span className="text-muted text-xs font-mono font-bold">◎</span>
        )}
      </span>
    );
  };

  return (
    <div className={`faceset-library rounded-xl bg-black/45 border border-white/10 overflow-hidden ${pickerOpen ? 'relative z-20' : ''}`}>
      {/* Header Bar */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 text-left select-none bg-surface/80 hover:bg-white/[0.04] transition-colors border-b border-white/5"
      >
        <span className="font-semibold text-micro uppercase tracking-[0.14em] text-white/60 flex items-center gap-2">
          <span>❖ FACESET LIBRARY</span>
        </span>
        <span className="flex items-center gap-2">
          {sources.length > 0 && (
            <span className="px-1.5 py-0.5 rounded-full bg-white/[0.06] text-nano text-white/60 border border-white/10">
              {sources.length}
            </span>
          )}
          <span className="text-muted text-xs">{open ? '▲' : '▼'}</span>
        </span>
      </button>

      {open && (
        <div className="p-3 space-y-2.5">
          {/* Action cluster buttons */}
          <div className="flex flex-wrap gap-1.5">
            {canSave && (
              <button
                type="button"
                className="button small primary-soft"
                disabled={busy}
                onClick={() => setSaveModalOpen(true)}
                title="Save currently selected face as a named FaceSet"
              >
                + Save face
              </button>
            )}
            <button
              type="button"
              className="button small"
              onClick={() => importRef.current?.click()}
              title="Import .fsz archive into local library"
            >
              ⬆ Import .fsz
            </button>
            <button
              type="button"
              className="button small"
              onClick={openFolder}
              title="Open FaceSets folder on disk"
            >
              Folder
            </button>
            <button
              type="button"
              className="button small"
              disabled={busy || sources.length === 0}
              onClick={rebuildThumbs}
              title="Rebuild previews, picking most frontal face in each set"
            >
              Fix thumbs
            </button>
            <button
              type="button"
              className="button small"
              onClick={refresh}
              title="Refresh FaceSet catalog"
            >
              ↻
            </button>
            <input
              ref={importRef}
              type="file"
              accept=".fsz"
              multiple
              className="hidden"
              style={{ display: 'none' }}
              onChange={handleImport}
            />
          </div>

          {sources.length === 0 ? (
            <p className="text-nano text-muted leading-relaxed">
              No saved FaceSets in library yet. Import a <span className="text-white/60">.fsz</span> file or build one from your target scene to keep it here across sessions.
            </p>
          ) : (
            <div>
              <div className="text-nano uppercase tracking-[0.12em] text-muted mb-1">Active FaceSet</div>
              <div className="relative" ref={pickerRef}>
                {/* Closed dropdown button */}
                <button
                  type="button"
                  onClick={() => setPickerOpen((v) => !v)}
                  className="w-full flex items-center justify-between gap-2 bg-black/40 border border-white/10 hover:border-white/25 rounded-lg px-2.5 py-1.5 text-left transition-colors"
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <Thumb entry={selected} size="w-6 h-6" />
                    <span className={`truncate text-note ${selected ? 'text-white/90 font-medium' : 'text-muted'}`}>
                      {selected ? selected.name : 'Select a FaceSet…'}
                    </span>
                  </span>
                  <span className={`text-muted text-xs shrink-0 transition-transform ${pickerOpen ? 'rotate-180' : ''}`}>
                    ⌄
                  </span>
                </button>

                {/* Dropdown Menu */}
                {pickerOpen && (
                  <div className="absolute z-50 left-0 right-0 mt-1 rounded-lg bg-[#14181a] border border-white/15 shadow-2xl overflow-hidden backdrop-blur-md">
                    {sources.length > 1 && (
                      <div className="p-1.5 border-b border-white/10 bg-black/20">
                        <input
                          aria-label="Search facesets"
                          value={filter}
                          onChange={(ev) => setFilter(ev.target.value)}
                          placeholder={`Search ${sources.length} facesets…`}
                          className="faceset-search-input w-full bg-black/50 border border-white/10 focus:border-[var(--accent)] rounded px-2 py-1 text-mini text-white/90 placeholder-white/30 outline-none"
                        />
                      </div>
                    )}
                    <div className="max-h-56 overflow-y-auto py-1 custom-scrollbar">
                      {shown.length === 0 ? (
                        <p className="text-mini text-muted py-2 text-center">No match for “{filter}”.</p>
                      ) : (
                        <VirtualList items={shown} itemKey={(e) => e.faceset_id} estimateHeight={42} gap={0} aria-label="FaceSets" renderItem={(e) => {
                          const isActive = selected?.faceset_id === e.faceset_id;
                          return (
                            <div
                              key={e.faceset_id}
                              className={`group flex items-center gap-2.5 px-2 py-1.5 mx-1 rounded-md cursor-pointer transition-colors ${
                                isActive ? 'bg-[var(--accent)]/15 border border-[var(--accent)]/30' : 'hover:bg-white/[0.06]'
                              }`}
                              onClick={() => {
                                if (renaming !== e.faceset_id) load(e);
                              }}
                            >
                              <Thumb entry={e} size="w-7 h-7" />
                              {renaming === e.faceset_id ? (
                                <input
                                  autoFocus
                                  aria-label="Rename FaceSet"
                                  value={renameVal}
                                  onClick={(ev) => ev.stopPropagation()}
                                  onChange={(ev) => setRenameVal(ev.target.value)}
                                  onBlur={() => commitRename(e)}
                                  onKeyDown={(ev) => {
                                    ev.stopPropagation();
                                    if (ev.key === 'Enter') commitRename(e);
                                    if (ev.key === 'Escape') cancelRename();
                                  }}
                                  className="flex-1 min-w-0 bg-black/60 border border-[var(--accent)] rounded px-1.5 py-0.5 text-mini text-white outline-none"
                                />
                              ) : (
                                <span className="flex-1 min-w-0 truncate text-note text-white/85" title={e.name}>
                                  {e.name}
                                  <span className="text-muted text-nano"> · {e.item_count} views</span>
                                </span>
                              )}

                              {/* Action buttons on hover/focus */}
                              <span className="flex items-center gap-1.5 text-mini text-muted shrink-0 opacity-80 group-hover:opacity-100 transition-opacity">
                                <button
                                  type="button"
                                  className="hover:text-white transition-colors px-1"
                                  title="Rename FaceSet"
                                  aria-label={`Rename ${e.name}`}
                                  onClick={(ev) => {
                                    ev.stopPropagation();
                                    beginRename(e);
                                  }}
                                >
                                  ✎
                                </button>
                                <a
                                  className="hover:text-white transition-colors px-1"
                                  title="Download .fsz"
                                  aria-label={`Download ${e.name}`}
                                  href={`/api/v1/facesets/${e.faceset_id}/download`}
                                  download={`${e.faceset_id}.fsz`}
                                  onClick={(ev) => ev.stopPropagation()}
                                >
                                  ⤓
                                </a>
                                <button
                                  type="button"
                                  className="hover:text-danger transition-colors px-1"
                                  title="Delete FaceSet"
                                  aria-label={`Delete ${e.name}`}
                                  onClick={(ev) => {
                                    ev.stopPropagation();
                                    void del(e);
                                  }}
                                >
                                  ✕
                                </button>
                              </span>
                            </div>
                          );
                        }} />
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Save Name Modal */}
      {saveModalOpen && (
        <div className="modal-backdrop" onClick={() => setSaveModalOpen(false)}>
          <div className="project-modal" style={{ maxWidth: '420px' }} onClick={(e) => e.stopPropagation()}>
            <div className="panel-heading">
              <div>
                <span className="eyebrow">SAVE TO LIBRARY</span>
                <h2>Name this FaceSet</h2>
              </div>
              <button className="icon-button" onClick={() => setSaveModalOpen(false)}>
                ×
              </button>
            </div>
            <p className="muted" style={{ margin: '12px 0' }}>
              Give this FaceSet identity a name. It will be stored in your persistent library and can be reloaded across projects without re-uploading.
            </p>
            <input
              autoFocus
              className="full"
              value={newFacesetName}
              placeholder="e.g. Scarlett — hero angles"
              onChange={(e) => setNewFacesetName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void handleSaveCurrent();
                if (e.key === 'Escape') setSaveModalOpen(false);
              }}
              style={{ padding: '8px 12px', fontSize: '12px' }}
            />
            <div className="flex gap-2" style={{ marginTop: '16px', display: 'flex', justifyContent: 'flex-end' }}>
              <button className="button" onClick={() => setSaveModalOpen(false)}>
                Cancel
              </button>
              <button
                className="button accent"
                disabled={!newFacesetName.trim() || busy}
                onClick={handleSaveCurrent}
              >
                Save FaceSet
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
