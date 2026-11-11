/** Any screen can ask the app shell to open the Model Hub without prop drilling. */
export const OPEN_MODEL_HUB = 'roop3d:open-model-hub';
export const openModelHub = (): void => { window.dispatchEvent(new Event(OPEN_MODEL_HUB)); };
