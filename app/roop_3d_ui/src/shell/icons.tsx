// Small stroke icon set (24px grid, currentColor). Decorative: the owning control carries the accessible name.
import type { ReactNode } from 'react';

const base = { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true, focusable: false } as const;
const make = (paths: ReactNode) => function Icon({ size = 18 }: { size?: number }) { return <svg {...base} width={size} height={size}>{paths}</svg>; };

export const IconHome = make(<><path d="M3 11l9-8 9 8" /><path d="M5 10v10h14V10" /><path d="M10 20v-6h4v6" /></>);
export const IconSwap = make(<><circle cx="9" cy="8" r="3.2" /><path d="M3 20c0-3.2 2.7-5.5 6-5.5" /><circle cx="16" cy="12" r="3.2" /><path d="M10.5 21c0-3 2.4-5.2 5.5-5.2s5.5 2.2 5.5 5.2" /></>);
export const IconLayers = make(<><path d="M12 3l9 5-9 5-9-5 9-5z" /><path d="M3 13l9 5 9-5" /></>);
export const IconActivity = make(<path d="M3 12h4l3-8 4 16 3-8h4" />);
export const IconUsers = make(<><circle cx="9" cy="8" r="3.5" /><path d="M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6" /><path d="M16 4.5a3.5 3.5 0 010 7" /><path d="M18 14.2c2.2.7 3.5 2.6 3.5 5.8" /></>);
export const IconWand = make(<><path d="M4 20L16 8" /><path d="M14 6l4 4" /><path d="M18 3v3M16.5 4.5h3M6 4v2M5 5h2" /></>);
export const IconFolder = make(<path d="M3 6.5A1.5 1.5 0 014.5 5H9l2 2.5h8.5A1.5 1.5 0 0121 9v9a1.5 1.5 0 01-1.5 1.5h-15A1.5 1.5 0 013 18V6.5z" />);
export const IconClock = make(<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>);
export const IconSliders = make(<><path d="M4 7h10M18 7h2M4 17h2M10 17h10" /><circle cx="16" cy="7" r="2" /><circle cx="8" cy="17" r="2" /></>);
export const IconSidebar = make(<><rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="M9 4v16" /></>);
export const IconSearch = make(<><circle cx="11" cy="11" r="6.5" /><path d="M20 20l-4-4" /></>);
export const IconSun = make(<><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19" /></>);
export const IconMoon = make(<path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z" />);
export const IconPalette = make(<><path d="M12 3a9 9 0 100 18c1.4 0 2-.9 2-1.8 0-1.2-1-1.6-1-2.7 0-1 .8-1.5 1.8-1.5H17a4 4 0 004-4c0-4.4-4-8-9-8z" /><circle cx="7.5" cy="11" r="1" /><circle cx="10" cy="7" r="1" /><circle cx="15" cy="7.5" r="1" /></>);
export const IconHelp = make(<><circle cx="12" cy="12" r="9" /><path d="M9.5 9.5a2.5 2.5 0 114 2c-.9.6-1.5 1.1-1.5 2.2" /><path d="M12 17.2v.1" /></>);
export const IconSave = make(<><path d="M5 3h11l3 3v15H5V3z" /><path d="M8 3v5h7V3M8 21v-7h8v7" /></>);
export const IconLibrary = make(<><rect x="4" y="4" width="4" height="16" rx="1" /><rect x="10" y="4" width="4" height="16" rx="1" /><path d="M17 6l3 .8-3.6 13.4-3-.8" /></>);
export const IconMinus = make(<path d="M6 12h12" />);
export const IconPlus = make(<path d="M12 6v12M6 12h12" />);
export const IconGauge = make(<><path d="M4.5 17a8.5 8.5 0 1115 0" /><path d="M12 17l4-5.5" /><path d="M12 8.5V7M7 10l-1-1M17 10l1-1" /></>);
export const IconCube = make(<><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" /><path d="M4 7.5l8 4.5 8-4.5M12 12v9" /></>);
