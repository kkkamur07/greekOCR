import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

const BASE = 'http://localhost:5173';
const OUT = path.resolve('docs/screenshots/ui-fixes-verification');

const PROJECT_ID = '2da5832f-8794-4d5f-8dd3-3ced5206b904';
const DRAFT_DOC_ID = '6d79ec94-6639-42ac-9d91-1336e8a7eeba';
const PUBLISHED_DOC_ID = 'ea494940-edf2-4e5b-b3cd-56d5db258ec0';
const PART_ID = 'f9a89184-495a-469f-aafa-e78daa2bd3dd';

async function login(page) {
  await page.goto(`${BASE}/login`);
  await page.getByLabel(/email/i).fill('dev@example.com');
  await page.getByLabel(/password/i).fill('dev-pass-123');
  await page.getByRole('button', { name: /sign in|log in/i }).click();
  await page.waitForURL(/\/(projects)?/, { timeout: 15000 });
}

async function shot(page, name, locator) {
  const file = path.join(OUT, `${name}.png`);
  if (locator) {
    await locator.screenshot({ path: file });
  } else {
    await page.screenshot({ path: file, fullPage: true });
  }
  console.log(`saved ${file}`);
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  await login(page);

  // 1. Draft badge alignment — document detail header
  await page.goto(`${BASE}/projects/${PROJECT_ID}/documents/${DRAFT_DOC_ID}`);
  await page.waitForSelector('.page-header__title-row .badge-draft', { timeout: 15000 });
  await shot(page, '01-draft-badge-document-header', page.locator('.page-header'));

  // Documents table status column
  await page.goto(`${BASE}/projects/${PROJECT_ID}`);
  await page.waitForSelector('.data-list .badge', { timeout: 15000 });
  await shot(page, '02-draft-badge-documents-table', page.locator('.data-panel').first());

  // 2. Transcription strip — ground truth only (no layer dropdown)
  await page.goto(
    `${BASE}/projects/${PROJECT_ID}/documents/${DRAFT_DOC_ID}/parts/${PART_ID}`,
  );
  await page.waitForSelector('.pe-canvas-wrap, .pe-toolbar', { timeout: 20000 });
  // Open transcription panel if hidden
  const txBtn = page.getByRole('button', { name: /transcription/i }).first();
  if (await txBtn.isVisible().catch(() => false)) {
    await txBtn.click();
  }
  await page.waitForSelector('.pe-strip', { timeout: 15000 });
  // Select first segment if any
  const segment = page.locator('.pe-strip__badge').first();
  await shot(page, '03-transcription-ground-truth-only', page.locator('.pe-strip'));

  // 3. Project jobs panel
  await page.goto(`${BASE}/projects/${PROJECT_ID}`);
  await page.waitForSelector('#project-jobs-heading', { timeout: 15000 });
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await shot(page, '04-project-jobs-panel', page.locator('.project-jobs-panel'));

  // 4. Public view — image with polygons + PDF toggle
  await page.goto(`${BASE}/public/projects/${PROJECT_ID}/documents/${PUBLISHED_DOC_ID}`);
  await page.waitForSelector('.public-page-canvas__image, .pub-canvas', { timeout: 20000 });
  await page.waitForTimeout(1500);
  await shot(page, '05-public-view-image-polygons', page.locator('.pub-split'));

  await page.getByRole('tab', { name: /transcription pdf/i }).click();
  await page.waitForTimeout(2000);
  await shot(page, '06-public-view-pdf-toggle', page.locator('.pub-split'));

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
