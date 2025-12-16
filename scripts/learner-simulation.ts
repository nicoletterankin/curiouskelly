import fs from 'node:fs';
import path from 'node:path';
import puppeteer, { type Browser, type Page } from 'puppeteer';

type TestResult = {
  name: string;
  passed: boolean;
  durationMs: number;
  error?: string;
  screenshotPath?: string;
};

type SimulationReport = {
  timestamp: string;
  baseUrl: string;
  headless: boolean;
  slowMo: number;
  total: number;
  passed: number;
  failed: number;
  durationMs: number;
  results: TestResult[];
  diagnostics: {
    consoleErrors: string[];
    pageErrors: string[];
    httpErrors: string[];
    requestFailures: string[];
  };
};

const BASE_URL = process.env.TEST_URL || 'http://localhost:3000';
const HEADLESS = (process.env.HEADLESS || '').toLowerCase() === 'true';
const SLOW_MO = Number.parseInt(process.env.SLOW_MO || '75', 10);
const OUTPUT_DIR = path.join('test-output', 'learner-simulation');
const START_LEARN_URL = `${BASE_URL}/learn.html?autoplay=false`;
const START_HOME_URL = `${BASE_URL}/`;

function ensureDir(p: string) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}

async function screenshot(page: Page, name: string) {
  ensureDir(OUTPUT_DIR);
  const file = path.join(OUTPUT_DIR, `${Date.now()}-${name}.png`);
  await page.screenshot({ path: file, fullPage: true });
  return file;
}

async function withTest(page: Page, name: string, fn: () => Promise<void>): Promise<TestResult> {
  const t0 = Date.now();
  try {
    await fn();
    return { name, passed: true, durationMs: Date.now() - t0 };
  } catch (e: any) {
    const shot = await screenshot(page, name.replace(/[^a-z0-9]+/gi, '_').toLowerCase());
    return {
      name,
      passed: false,
      durationMs: Date.now() - t0,
      error: e?.message || String(e),
      screenshotPath: shot,
    };
  }
}

async function clearAppStorage(page: Page) {
  await page.evaluate(() => {
    try {
      localStorage.clear();
      sessionStorage.clear();
    } catch (_) {}
  });
}

async function waitForActiveScene(page: Page, sceneId: string, timeoutMs = 20000) {
  await page.waitForFunction(
    (id) => {
      const el = document.getElementById(id);
      return !!el && el.classList.contains('active');
    },
    { timeout: timeoutMs },
    sceneId
  );
}

async function ensureLessonLoaded(page: Page) {
  await waitForActiveScene(page, 'scene-lesson', 30000);
  await page.waitForFunction(
    () => {
      const topic = document.getElementById('lesson-topic')?.textContent || '';
      const bar = document.getElementById('phase-bar');
      const caption = document.getElementById('caption-text')?.textContent || '';
      return !!bar && topic.trim().length > 0 && !topic.includes('Loading') && caption.trim().length > 10;
    },
    { timeout: 30000 }
  );
}

async function clickIfExists(page: Page, selector: string) {
  const el = await page.$(selector);
  if (!el) return false;
  await el.click();
  return true;
}

async function openLeftPanel(page: Page) {
  // Prefer in-app function if present; fallback to any obvious chat button later.
  await page.evaluate(function () {
    try {
      // @ts-ignore
      if (typeof openPanel === 'function') openPanel('left');
    } catch (_) {}
  });
  await page.waitForFunction(() => document.documentElement.getAttribute('data-panel-open') === 'left', { timeout: 10000 });
}

async function assertClickable(page: Page, selector: string, timeoutMs = 10000) {
  await page.waitForSelector(selector, { timeout: timeoutMs });
  await page.evaluate(function (sel) {
    const el = document.querySelector(sel) as HTMLElement | null;
    if (!el) throw new Error(`missing:${sel}`);
    el.scrollIntoView({ block: 'center', inline: 'center' });
  }, selector);

  await page.waitForFunction(
    (sel) => {
      const el = document.querySelector(sel) as HTMLElement | null;
      if (!el) return false;
      const r = el.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) return false;
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      const at = document.elementFromPoint(cx, cy);
      return !!at && (at === el || el.contains(at));
    },
    { timeout: timeoutMs },
    selector
  );
}

async function clickFirstMatch(page: Page, selectors: string[], predicate: (text: string, elTag: string) => boolean) {
  for (const sel of selectors) {
    const els = await page.$$(sel);
    for (const el of els) {
      const info = await el.evaluate(function (node) {
        const text = (node.textContent || '').trim();
        const tag = (node as HTMLElement).tagName || '';
        return { text, tag };
      });
      if (predicate(info.text, info.tag)) {
        await el.click();
        return true;
      }
    }
  }
  return false;
}

async function runLessonSmoke(page: Page, results: TestResult[]) {
  results.push(
    await withTest(page, 'Lesson: load learn.html (today)', async () => {
      await page.goto(START_LEARN_URL, { waitUntil: 'networkidle2' });
      await clearAppStorage(page);
      await page.reload({ waitUntil: 'networkidle2' });
      await waitForActiveScene(page, 'scene-character', 30000);
    })
  );

  results.push(
    await withTest(page, 'Lesson: onboarding -> lesson', async () => {
      await page.waitForSelector('#btn-start', { timeout: 15000 });
      await page.click('#btn-start');
      await ensureLessonLoaded(page);
    })
  );

  results.push(
    await withTest(page, 'Lesson: cliff choice advances', async () => {
      // Jump to cliff if helper exists, otherwise tap next until it appears.
      await page.evaluate(function () {
        try {
          // @ts-ignore
          if (typeof state !== 'undefined') {
            // Reset flags and jump to cliff
            // @ts-ignore
            state.waitingForChoice = false;
            // @ts-ignore
            state.currentPhase = 1;
            // @ts-ignore
            if (typeof updatePhaseProgress === 'function') updatePhaseProgress();
            // @ts-ignore
            if (typeof saveState === 'function') saveState();
          }
        } catch (_) {}
      });

      await page.waitForFunction(
        () => {
          const el = document.getElementById('cliff-container');
          if (!el) return false;
          const hidden = el.hasAttribute('hidden') || el.classList.contains('hidden');
          return !hidden;
        },
        { timeout: 20000 }
      );

      await page.click('#cliff-choice-a');

      // Verify state reflects choice and moved beyond cliff
      await page.waitForFunction(
        () => {
          // @ts-ignore
          return typeof state !== 'undefined' && (state.cliffChoice === 'A' || state.currentPhase >= 2);
        },
        { timeout: 20000 }
      );
    })
  );

  results.push(
    await withTest(page, 'Lesson: chat input reachable', async () => {
      await ensureLessonLoaded(page);
      await openLeftPanel(page);
      await assertClickable(page, '#kelly-chat-input', 10000);
      await page.focus('#kelly-chat-input');
      await page.keyboard.type('Hello Kelly');
      await page.keyboard.press('Enter');
    })
  );
}

async function runMarketingToCheckout(page: Page, results: TestResult[]) {
  results.push(
    await withTest(page, 'Marketing: home loads', async () => {
      await page.goto(START_HOME_URL, { waitUntil: 'networkidle2' });
      // Basic sanity: at least some page content.
      const title = await page.title();
      if (!title || title.trim().length === 0) throw new Error('missing title');
    })
  );

  results.push(
    await withTest(page, 'Marketing: navigate to pricing', async () => {
      // Try to click an in-page pricing link; fallback to direct.
      const clicked = await clickFirstMatch(
        page,
        ['a[href]'],
        (text, _tag) => text.toLowerCase().includes('pricing')
      );

      if (clicked) await page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 20000 }).catch(() => {});
      if (!clicked) await page.goto(`${BASE_URL}/pricing.html`, { waitUntil: 'networkidle2' });

      const url = page.url();
      if (!url.includes('pricing')) {
        // If routing uses /pricing (no extension), accept that too.
        if (!/\/pricing(\.html)?(\?|#|$)/.test(new URL(url).pathname)) {
          throw new Error(`expected pricing page, got ${url}`);
        }
      }
    })
  );

  results.push(
    await withTest(page, 'Pricing: start checkout (some navigation happens)', async () => {
      // Ensure pricing scripts are ready before clicking. Without this, a fast click can happen
      // before module scripts finish defining `window.handleCheckout`.
      await page.waitForFunction(
        function () {
          // @ts-ignore
          return typeof window.handleCheckout === 'function';
        },
        { timeout: 15000 }
      );

      // Some implementations open a new window for Stripe; capture popups.
      let popupUrl: string | null = null;
      (page as any).once('popup', (p: Page) => {
        popupUrl = p.url();
      });

      // Capture/dismiss dialogs (alert can block navigation in headless).
      let lastDialog: string | null = null;
      page.once('dialog', async (d) => {
        lastDialog = d.message();
        await d.dismiss();
      });

      const before = page.url();

      // Click a likely checkout CTA.
      const clicked = await clickFirstMatch(page, ['button', 'a'], (text, tag) => {
        const t = text.toLowerCase();
        if (t.includes('subscribe monthly')) return true;
        if (t === 'subscribe') return true;
        if (t.includes('start learning')) return true;
        // Some buttons might be “Monthly” only.
        if (tag.toUpperCase() === 'BUTTON' && t.includes('monthly') && (t.includes('subscribe') || t.includes('join') || t.includes('unlock'))) {
          return true;
        }
        return false;
      });

      if (!clicked) throw new Error('could not find a subscribe/checkout CTA');

      // Wait for navigation/popup/dialog.
      await page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 10000 }).catch(() => {});
      await new Promise((r) => setTimeout(r, 500));

      const after = page.url();
      const moved = after !== before;

      if (!moved && !popupUrl) {
        if (lastDialog) throw new Error(`checkout blocked by dialog: ${lastDialog}`);
        throw new Error('click did not navigate or open a checkout window');
      }

      // If we navigated to a known login gate, that still counts as "something happened".
      // If we ended up on learn.html, also ok.
    })
  );
}

async function main() {
  ensureDir(OUTPUT_DIR);

  const browser: Browser = await puppeteer.launch({
    headless: HEADLESS ? 'new' : false,
    slowMo: SLOW_MO,
    args: ['--window-size=1280,720'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });

  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const httpErrors: string[] = [];
  const requestFailures: string[] = [];

  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => pageErrors.push(err?.message || String(err)));
  page.on('response', (resp) => {
    try {
      const status = resp.status();
      if (status >= 400) httpErrors.push(`${status} ${resp.url()}`);
    } catch (_) {}
  });
  page.on('requestfailed', (req) => {
    try {
      const failure = req.failure();
      requestFailures.push(`${failure?.errorText || 'request_failed'} ${req.method()} ${req.url()}`);
    } catch (_) {}
  });

  const startSuite = Date.now();
  const results: TestResult[] = [];

  await runLessonSmoke(page, results);
  await runMarketingToCheckout(page, results);

  await browser.close();

  const passed = results.filter((r) => r.passed).length;
  const report: SimulationReport = {
    timestamp: new Date().toISOString(),
    baseUrl: BASE_URL,
    headless: HEADLESS,
    slowMo: SLOW_MO,
    total: results.length,
    passed,
    failed: results.length - passed,
    durationMs: Date.now() - startSuite,
    results,
    diagnostics: { consoleErrors, pageErrors, httpErrors, requestFailures },
  };

  const reportPath = path.join(OUTPUT_DIR, 'simulation-report.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

  // Keep output copy/paste friendly
  // eslint-disable-next-line no-console
  console.log('LEARNER SIMULATION TEST SUITE');
  // eslint-disable-next-line no-console
  console.log('============================');
  // eslint-disable-next-line no-console
  console.log(`Base URL: ${BASE_URL}`);
  // eslint-disable-next-line no-console
  console.log(`Total: ${report.total}`);
  // eslint-disable-next-line no-console
  console.log(`Passed: ${report.passed}`);
  // eslint-disable-next-line no-console
  console.log(`Failed: ${report.failed}`);
  // eslint-disable-next-line no-console
  console.log(`Duration: ${(report.durationMs / 1000).toFixed(1)}s`);
  // eslint-disable-next-line no-console
  console.log('');

  for (const r of report.results) {
    // eslint-disable-next-line no-console
    console.log(`${r.passed ? 'PASS' : 'FAIL'}: ${r.name} (${r.durationMs}ms)${r.error ? ` - ${r.error}` : ''}${r.screenshotPath ? ` [${r.screenshotPath}]` : ''}`);
  }

  // eslint-disable-next-line no-console
  console.log('');
  // eslint-disable-next-line no-console
  console.log(`Console errors: ${consoleErrors.length}`);
  // eslint-disable-next-line no-console
  console.log(`Page errors: ${pageErrors.length}`);
  // eslint-disable-next-line no-console
  console.log(`HTTP errors (>=400): ${httpErrors.length}`);
  // eslint-disable-next-line no-console
  console.log(`Request failures: ${requestFailures.length}`);
  // eslint-disable-next-line no-console
  console.log(`Report saved: ${reportPath}`);

  process.exit(report.failed > 0 ? 1 : 0);
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
