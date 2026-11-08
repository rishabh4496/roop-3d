import { expect, test } from '@playwright/test';
import { boot } from './helpers/mockApi';

/**
 * Final workstation stage, in a real browser: the 3x3 sweep with the Enhancer and SH-relighting axes and native-pixel zoom, and the
 * Stream Overlap & PCIe tab fed by a payload shaped exactly like GET /api/v1/system/pipeline.
 */
const PIPELINE = {
  pipelines: [{
    id: 'p1', name: 'render', running: true, frames_out: 48, fps: 24.6, bottleneck: 'swap', wall_s: 2,
    stages: [], pools: [{ backend: 'cuda_pinned', count: 10, in_flight: 4, bytes: 25_000_000, waits: 3, blocked_s: 0.42 }],
    lanes: { window_ms: 160, stages: {
      decode: [[40, 0, 14], [41, 14, 30], [42, 30, 47], [43, 47, 62]],
      track: [[40, 14, 50], [41, 50, 84], [42, 84, 118], [43, 118, 150]],
      swap: [[40, 50, 96], [41, 96, 142]],
      composite: [[40, 96, 140], [41, 142, 160]],
      write: [[40, 140, 150]],
    } },
    overlap_wall: { ge1: 1, ge2: 0.71, ge3: 0.34, mean_active: 2.1 }, overlap_gpu_streams: null,
    gpu_stream_note: 'No CUDA stream windows were recorded: torch.cuda streams need PyTorch, which this environment does not use, so only host-thread overlap is measured.',
    transfers: { measured: true, scope: 'onnxruntime-boundary', installed_on: '1.23.2', gpu_runs: 96,
      total: { h2d_count: 96, h2d_bytes: 96 * 786432, d2h_count: 96, d2h_bytes: 96 * 786432, intermediate_copies: 192, boundary_copies: 0 },
      per_frame: { h2d_tensors: 2, d2h_tensors: 2, h2d_bytes: 1572864, d2h_bytes: 1572864, intermediate_copies: 4 },
      by_stage: { track: { h2d_count: 48, d2h_count: 48, h2d_bytes: 48 * 786432, d2h_bytes: 48 * 786432 }, swap: { h2d_count: 48, d2h_count: 48, h2d_bytes: 48 * 786432, d2h_bytes: 48 * 786432 } },
      not_counted: ['torch / CV-CUDA / OpenCV-CUDA copies', 'NVENC / NVDEC internal transfers', 'copies inside ONNX Runtime between graph partitions'],
      zero_intermediate_copies: false },
  }],
  hardware: { backend: 'nvml', latest: { gpu_util_pct: 71, pcie_tx_mb_s: 812, pcie_rx_mb_s: 640, vram_used_mib: 5120, vram_total_mib: 12282 }, window: {} },
  pcie_note: 'NVML reports PCIe throughput over a ~20 ms window; values are instantaneous, not averaged.',
};

test('the stream overlap tab draws the measured lanes, the copy verdict and the ring buffer', async ({ page }) => {
  const { errors } = await boot(page, { route: '/#/processing' });
  // Registered after boot(): the most recently added route wins, and boot() installs a catch-all for /api/v1/**.
  await page.route('**/api/v1/system/pipeline', route => route.fulfill({ json: PIPELINE }));
  await page.getByRole('button', { name: 'Stream Overlap & PCIe' }).first().click();
  const panel = page.getByTestId('stream-panel').first();
  await expect(panel.getByTestId('lane-swap')).toContainText('Stage 2 · 3D swap');
  await expect(panel.getByTestId('lane-composite')).toContainText('UltraMax');
  await expect(panel.getByTestId('overlap-line')).toContainText('71%');
  await expect(panel.getByTestId('gpu-stream-note')).toContainText('not measured');
  await expect(panel.getByTestId('transfer-verdict')).toContainText('4 tensor copies per frame');
  await expect(panel.getByTestId('transfer-verdict')).toContainText('not zero-copy');
  await expect(panel.getByTestId('pool-row')).toContainText('4/10 slots in flight');
  // every lane draws real segments, positioned on the shared axis
  expect(await panel.locator('.lane-seg').count()).toBe(4 + 4 + 2 + 2 + 1);
  await panel.scrollIntoViewIfNeeded();
  await panel.screenshot({ path: 'test-results/final-stream-overlap.png' });
  expect(errors).toEqual([]);
});

test('the sweep compares enhancers across relighting strengths and zooms a tile to native pixels', async ({ page }) => {
  const { seen, errors } = await boot(page, { openProject: true });
  await page.getByRole('button', { name: '3×3 Sweep' }).first().click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('X axis setting').selectOption('enhancer');
  await dialog.getByLabel('Y axis setting').selectOption('relight_strength');
  await expect(dialog).toContainText('None · GPEN Realistic · UltraMax');
  await expect(dialog).toContainText('25% · 60% · 100%');
  await dialog.getByRole('button', { name: 'Generate contact sheet' }).click();
  await expect(dialog.locator('.sweep-cell.ready')).toHaveCount(9);
  // the nine requests are the real previews: restorer and relighting differ per tile
  const payloads = seen.previews.slice(-9);
  expect(payloads.map(p => [p.restorer, p.relight ? p.relight_strength : null])).toEqual([
    ['off', .25], ['gpen', .25], ['ultramax', .25], ['off', .6], ['gpen', .6], ['ultramax', .6], ['off', 1], ['gpen', 1], ['ultramax', 1]]);
  await page.screenshot({ path: 'test-results/final-sweep-sheet.png' });

  await dialog.getByRole('button', { name: 'Zoom tile 9' }).click();
  const frame = dialog.getByTestId('zoom-frame');
  await expect(frame).toHaveAttribute('data-zoom', '1.00');
  await dialog.getByRole('button', { name: '400%' }).click();
  await expect(frame).toHaveAttribute('data-zoom', '4.00');
  await frame.hover();
  await page.mouse.wheel(0, -300);
  expect(Number(await frame.getAttribute('data-zoom'))).toBeGreaterThan(4);
  await page.screenshot({ path: 'test-results/final-sweep-zoom.png' });
  await dialog.getByRole('button', { name: 'Apply these settings' }).click();
  await expect(dialog).toBeHidden();
  expect(errors).toEqual([]);
});
