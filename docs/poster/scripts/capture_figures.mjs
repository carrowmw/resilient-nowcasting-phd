// Capture poster-quality PNGs of the presentation's d3 figures.
// Usage: node docs/poster/scripts/capture_figures.mjs
// Requires: an HTTP server on :8321 serving /workspace/presentation
//           (d3.json() cannot fetch figure data over file://)
import puppeteer from '/workspace/presentation/node_modules/puppeteer/lib/esm/puppeteer/puppeteer.js';

const BASE = 'http://localhost:8321/figures';
const OUT = '/workspace/docs/poster';

// Per-figure: which chrome to strip (poster cards supply their own titles),
// which element proves the d3 render finished, and what to screenshot.
const FIGURES = [
  {
    name: 'zinb_confidence',
    url: `${BASE}/zinb-confidence-poster.html`,
    waitFor: '#chart svg',
    hideCss: `
      .figure-title, .figure-subtitle, .sensor-tabs, .tooltip { display: none !important; }
      .figure-header { border-bottom: none !important; padding-bottom: 0 !important; margin-bottom: 8px !important; justify-content: flex-end !important; }
      body { height: auto !important; }
      .figure-container { padding: 24px !important; height: auto !important; }
      #chart { flex: none !important; }
    `,
    clip: '.figure-container',
  },
  {
    name: 'pit_underdispersed',
    url: `${BASE}/pit-underdispersed.html`,
    waitFor: '#diagram svg',
    hideCss: `
      .figure-header { display: none !important; }
      .figure-container { padding: 20px !important; }
    `,
    clip: '.figure-container',
  },
  {
    name: 'pit_calibrated',
    url: `${BASE}/pit-calibrated.html`,
    waitFor: '#diagram svg',
    hideCss: `
      .figure-header { display: none !important; }
      .figure-container { padding: 20px !important; }
    `,
    clip: '.figure-container',
  },
];

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/chromium',
  args: ['--no-sandbox'],
});
try {
  for (const fig of FIGURES) {
    const page = await browser.newPage();
    await page.setViewport({ width: 1700, height: 1100, deviceScaleFactor: 2 });
    await page.goto(fig.url, { waitUntil: 'networkidle0' });
    await page.waitForSelector(fig.waitFor, { timeout: 15000 });
    await page.addStyleTag({ content: fig.hideCss });
    await page.evaluate(() => document.fonts.ready);
    await new Promise((r) => setTimeout(r, 800));
    const el = await page.$(fig.clip);
    await el.screenshot({ path: `${OUT}/${fig.name}.png` });
    const box = await el.boundingBox();
    console.log(`${fig.name}.png  ${Math.round(box.width)}x${Math.round(box.height)} css px (@2x)`);
    await page.close();
  }
} finally {
  await browser.close();
}
