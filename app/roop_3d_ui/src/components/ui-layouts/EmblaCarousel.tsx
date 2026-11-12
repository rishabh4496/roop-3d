import { useCallback, useEffect, useState, type HTMLAttributes, type ReactNode } from 'react';
import useEmblaCarousel from 'embla-carousel-react';
import type { EmblaCarouselType, EmblaOptionsType } from 'embla-carousel';
import { useReducedMotion } from 'framer-motion';
import { cn } from '../../lib/utils';

/**
 * Thin, accessible wrapper over Embla (ui-layouts "Carousel"): draggable with inertia, snap-free scrolling, previous / next buttons
 * and a way to bring a slide into view. The wrapper owns the viewport + container markup; slides are the children.
 */
export interface EmblaCarouselProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children'> {
  children?: ReactNode;
  options?: EmblaOptionsType;
  /** Accessible name of the whole carousel. */
  label: string;
  /** Slide to keep in view (e.g. the one under the playhead). It only scrolls when that slide is not already visible. */
  followIndex?: number | null;
  /** Called once with the Embla API when it is ready. */
  onApi?: (api: EmblaCarouselType) => void;
  showButtons?: boolean;
  /** Class for the slide strip (`display:flex`). */
  trackClassName?: string;
}

export const DEFAULT_OPTIONS: EmblaOptionsType = { dragFree: true, containScroll: 'trimSnaps', align: 'start', axis: 'x' };

/**
 * Embla maps over `window.matchMedia` unconditionally, which throws where it does not exist (old webviews, jsdom). There the strip is a plain
 * horizontally scrollable row: same markup and classes, native scrolling instead of drag inertia.
 */
export function EmblaCarousel(props: EmblaCarouselProps) {
  const supported = typeof window !== 'undefined' && typeof window.matchMedia === 'function';
  return supported ? <EmblaDriven {...props} /> : <PlainStrip {...props} />;
}

function PlainStrip({ children, label, className, trackClassName, options: _options, followIndex: _follow, onApi: _api, showButtons: _buttons, ...rest }: EmblaCarouselProps) {
  return (
    <div {...rest} data-scroll className={cn('ul-embla ul-embla--plain', className)} role="region" aria-roledescription="carousel" aria-label={label}>
      <div className="ul-embla__viewport ul-embla__viewport--plain" tabIndex={0}><div className={cn('ul-embla__track', trackClassName)}>{children}</div></div>
    </div>
  );
}

function EmblaDriven({ children, options, label, followIndex = null, onApi, showButtons = true, className, trackClassName, ...rest }: EmblaCarouselProps) {
  const reduce = useReducedMotion();
  const [ref, api] = useEmblaCarousel({ ...DEFAULT_OPTIONS, ...options });
  const [canPrev, setCanPrev] = useState(false), [canNext, setCanNext] = useState(false);

  useEffect(() => { if (api) onApi?.(api); }, [api, onApi]);
  useEffect(() => {
    if (!api) return undefined;
    const sync = () => { setCanPrev(api.canScrollPrev()); setCanNext(api.canScrollNext()); };
    sync();
    api.on('select', sync).on('scroll', sync).on('reInit', sync);
    return () => { api.off('select', sync).off('scroll', sync).off('reInit', sync); };
  }, [api]);

  useEffect(() => {
    if (!api || followIndex === null || followIndex < 0) return;
    if (api.slidesInView().includes(followIndex)) return;       // already visible: never fight the user's own drag
    api.scrollTo(followIndex, !!reduce);
  }, [api, followIndex, reduce]);

  const step = useCallback((direction: -1 | 1) => { if (direction < 0) api?.scrollPrev(!!reduce); else api?.scrollNext(!!reduce); }, [api, reduce]);

  return (
    <div {...rest} data-scroll className={cn('ul-embla', className)} role="region" aria-roledescription="carousel" aria-label={label}>
      {showButtons && <button type="button" className="ul-embla__btn ul-embla__btn--prev" aria-label="Scroll left" disabled={!canPrev} onClick={() => step(-1)}><span aria-hidden>‹</span></button>}
      <div className="ul-embla__viewport" ref={ref}>
        <div className={cn('ul-embla__track', trackClassName)}>{children}</div>
      </div>
      {showButtons && <button type="button" className="ul-embla__btn ul-embla__btn--next" aria-label="Scroll right" disabled={!canNext} onClick={() => step(1)}><span aria-hidden>›</span></button>}
    </div>
  );
}
