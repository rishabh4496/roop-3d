import { createAtomicStore, useStoreSelector } from './atomicStore';

// Selection is captured in each saved job. Older jobs default to the reference adapter.
export const engineSelection = createAtomicStore({ engineId: 'reference_onnx' });
export const selectEngine = (engineId: string) => engineSelection.set({ engineId });
export const useSelectedEngine = () => useStoreSelector(engineSelection, state => state.engineId);
