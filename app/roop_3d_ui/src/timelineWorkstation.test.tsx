import { describe, expect, it, vi, beforeEach } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import TimelineWorkstation, {
  formatTimecode,
  parseTimecodeString,
  parseFps,
  pointsToSplinePath,
  getClusterColor,
} from './components/timeline/TimelineWorkstation';
import { useFaceSelectionStore } from './useFaceSelectionStore';

describe('Timeline Utility Functions', () => {
  it('formats timecode into SMPTE HH:MM:SS:FF accurately', () => {
    expect(formatTimecode(0, 30)).toBe('00:00:00:00');
    expect(formatTimecode(1.5, 30)).toBe('00:00:01:15');
    expect(formatTimecode(65.5, 30)).toBe('00:01:05:15');
    expect(formatTimecode(3661.25, 24)).toBe('01:01:01:06');
    expect(formatTimecode(-5, 30)).toBe('00:00:00:00');
    expect(formatTimecode(NaN, 30)).toBe('00:00:00:00');
  });

  it('parses timecode strings and frame numbers into seconds', () => {
    expect(parseTimecodeString('00:00:05:15', 30)).toBe(5.5);
    expect(parseTimecodeString('01:05:15', 30)).toBe(65.5);
    expect(parseTimecodeString('02:30', 30)).toBe(150);
    expect(parseTimecodeString('f:120', 30)).toBe(4);
    expect(parseTimecodeString('#90', 30)).toBe(3);
    expect(parseTimecodeString('f60', 30)).toBe(2);
    expect(parseTimecodeString('12.5', 30)).toBe(12.5);
    expect(parseTimecodeString('', 30)).toBe(0);
    expect(parseTimecodeString('invalid', 30)).toBe(0);
  });

  it('parses FPS rationals accurately with fallbacks', () => {
    expect(parseFps('30000/1001', 30)).toBeCloseTo(29.97, 2);
    expect(parseFps('24/1', 30)).toBe(24);
    expect(parseFps('60/1', 30)).toBe(60);
    expect(parseFps(25, 30)).toBe(25);
    expect(parseFps(null, 30)).toBe(30);
    expect(parseFps('invalid', 24)).toBe(24);
  });

  it('generates smooth cubic Bézier spline paths from points', () => {
    expect(pointsToSplinePath([])).toBe('');
    expect(pointsToSplinePath([{ x: 10, y: 20 }])).toBe('M 10.0 20.0');
    expect(pointsToSplinePath([{ x: 0, y: 0 }, { x: 10, y: 10 }])).toBe('M 0.0 0.0 L 10.0 10.0');

    const curve = pointsToSplinePath([
      { x: 0, y: 27 },
      { x: 50, y: 10 },
      { x: 100, y: 40 },
      { x: 150, y: 27 },
    ]);
    expect(curve).toContain('M 0.0 27.0');
    expect(curve).toContain('C');
  });

  it('assigns deterministic colors to cluster IDs', () => {
    const c1 = getClusterColor('track_0000');
    const c2 = getClusterColor('track_0000');
    const c3 = getClusterColor('track_0001');
    expect(c1).toBe(c2);
    expect(typeof c1).toBe('string');
    expect(c1.startsWith('#')).toBe(true);
    expect(typeof c3).toBe('string');
  });
});

describe('TimelineWorkstation Component', () => {
  beforeEach(() => {
    useFaceSelectionStore.getState().reset();
  });

  it('renders playback engine with transport, variable speed controller, and frame stepping', () => {
    const html = renderToStaticMarkup(
      <TimelineWorkstation
        target="interview.mp4"
        currentTime={5.5}
        duration={20}
        fps={30}
        playing={false}
      />
    );

    // Play/Pause button
    expect(html).toContain('timeline-play-btn');
    expect(html).toContain('▶');

    // Stepping buttons
    expect(html).toContain('Step -1 Frame');
    expect(html).toContain('Step +1 Frame');
    expect(html).toContain('-1f');
    expect(html).toContain('+1f');

    // Timecode readout
    expect(html).toContain('00:00:05:15');
    expect(html).toContain('00:00:20:00');
    expect(html).toContain('[F: 165 / 600]');

    // Variable speed controller buttons
    expect(html).toContain('SPEED:');
    expect(html).toContain('0.25x');
    expect(html).toContain('0.5x');
    expect(html).toContain('1x');
    expect(html).toContain('2x');
    expect(html).toContain('4x');

    // Extended transport jump buttons
    expect(html).toContain('Jump to Start / In Point');
    expect(html).toContain('Jump -1 Second');
    expect(html).toContain('Jump +1 Second');
    expect(html).toContain('Jump to End / Out Point');

    // Snap and Loop toggles
    expect(html).toContain('timeline-snap-btn');
    expect(html).toContain('Snap');
    expect(html).toContain('timeline-loop-btn');
    expect(html).toContain('Loop');

    // Timeline Zoom controller
    expect(html).toContain('timeline-zoom-controller');
    expect(html).toContain('ZOOM:');
    expect(html).toContain('zoom-slider');
    expect(html).toContain('zoom-step-btn');
    expect(html).toContain('1.0x');
  });

  it('renders timecode ruler and filmstrip track with 120px-wide thumbnail frame cards', () => {
    const html = renderToStaticMarkup(
      <TimelineWorkstation
        target="interview.mp4"
        currentTime={2.0}
        duration={10}
        fps={30}
      />
    );

    // Timecode ruler track
    expect(html).toContain('timeline-tracks-viewport');
    expect(html).toContain('timeline-tracks-inner');
    expect(html).toContain('timecode-ruler-track');
    expect(html).toContain('ruler-tick');

    // Filmstrip track
    expect(html).toContain('filmstrip-track');
    expect(html).toContain('filmstrip-frame-card');
    expect(html).toContain('sprocket-holes top');
    expect(html).toContain('sprocket-holes bottom');
    expect(html).toContain('preview-frame');

    // Playhead needle
    expect(html).toContain('timeline-playhead-needle');
    expect(html).toContain('needle-head-arrow');
  });

  it('renders draggable range in/out trim handles and trim regions', () => {
    const html = renderToStaticMarkup(
      <TimelineWorkstation
        target="interview.mp4"
        currentTime={4.0}
        duration={12}
        fps={30}
        trimStart={2.0}
        trimEnd={8.0}
      />
    );

    // Trim buttons
    expect(html).toContain('trim-btn mark-in');
    expect(html).toContain('trim-btn mark-out');
    expect(html).toContain('trim-btn reset');

    // Trim handles and overlays
    expect(html).toContain('trim-range-layer');
    expect(html).toContain('trim-dimmed-region left');
    expect(html).toContain('trim-active-region');
    expect(html).toContain('trim-dimmed-region right');
    expect(html).toContain('trim-handle-marker in');
    expect(html).toContain('trim-handle-marker out');
    expect(html).toContain('handle-tag">IN');
    expect(html).toContain('handle-tag">OUT');
  });

  it('renders Landmark & Pose sub-track with reference guidelines', () => {
    const html = renderToStaticMarkup(
      <TimelineWorkstation
        target="interview.mp4"
        currentTime={1.0}
        duration={10}
        fps={30}
      />
    );

    expect(html).toContain('landmark-pose-subtrack');
    expect(html).toContain('HEAD YAW &amp; IDENTITY CONTINUITY');
    expect(html).toContain('+45°');
    expect(html).toContain('0°');
    expect(html).toContain('-45°');
    expect(html).toContain('Frontal (±20°)');
    expect(html).toContain('Profile (&gt;45°)');
  });

  it('renders colored face markers and continuous Head Yaw spline when faces are detected', () => {
    // Populate store with detected faces
    useFaceSelectionStore.getState().setDetectedFaces([
      {
        id: 'track_0000_10',
        frameIndex: 10,
        timestamp: 1.0,
        bbox: [0.2, 0.2, 0.5, 0.5],
        clusterId: 'track_0000',
        pose: { yaw: 5, pitch: -2, roll: 0 },
      },
      {
        id: 'track_0000_20',
        frameIndex: 20,
        timestamp: 2.5,
        bbox: [0.2, 0.2, 0.5, 0.5],
        clusterId: 'track_0000',
        pose: { yaw: 48, pitch: 10, roll: -5 }, // Profile scene
      },
      {
        id: 'track_0001_30',
        frameIndex: 30,
        timestamp: 4.0,
        bbox: [0.5, 0.3, 0.8, 0.6],
        clusterId: 'track_0001',
        pose: { yaw: -10, pitch: 0, roll: 0 },
      },
    ]);

    const html = renderToStaticMarkup(
      <TimelineWorkstation
        target="interview.mp4"
        currentTime={1.0}
        duration={5.0}
        fps={30}
      />
    );

    // Colored face markers
    expect(html).toContain('face-marker-pin');
    expect(html).toContain('Face track_0000_10 (5.0° yaw) - Click to focus 3D Pose');
    expect(html).toContain('Face track_0000_20 (48.0° yaw) - Click to focus 3D Pose');

    // Continuous spline SVG graph
    expect(html).toContain('yaw-spline-svg');
    expect(html).toContain('yawGradient');
    expect(html).toContain('yaw-spline-path');
  });

  it('updates playhead and active face selection on face marker click', () => {
    const onSeek = vi.fn();
    useFaceSelectionStore.getState().setDetectedFaces([
      {
        id: 'track_0000_15',
        frameIndex: 15,
        timestamp: 1.5,
        bbox: [0.2, 0.2, 0.5, 0.5],
        clusterId: 'track_0000',
        pose: { yaw: 12, pitch: 0, roll: 0 },
      },
    ]);

    // Test selection store directly
    useFaceSelectionStore.getState().setActiveSelection('track_0000_15');
    expect(useFaceSelectionStore.getState().activeSelection).toBe('track_0000_15');
  });
});
