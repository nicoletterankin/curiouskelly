/*
  Proof harness for KellyLessonLoader baseline.

  This runs in Node, stubbing:
  - window/document
  - localStorage
  - fetch() for /lessons/day-<N>.json using local filesystem

  It prints the returned payload from getLesson() for:
  1) Day 1 seed (happy path)
  2) ensureMvpLessonShape() applied to an intentionally-incomplete "Supabase-like" payload
  3) On-demand topic "volcanoes" in Grow track (localStorage)
*/

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');

// -----------------------------
// Minimal browser stubs
// -----------------------------
class LocalStorageStub {
  constructor() { this._m = new Map(); }
  getItem(k) { return this._m.has(String(k)) ? this._m.get(String(k)) : null; }
  setItem(k, v) { this._m.set(String(k), String(v)); }
  removeItem(k) { this._m.delete(String(k)); }
  clear() { this._m.clear(); }
}

global.window = {};
global.document = {
  readyState: 'complete',
  getElementById: () => null,
  addEventListener: () => {},
};
global.MutationObserver = class { observe() {} disconnect() {} };

global.localStorage = new LocalStorageStub();

global.location = { search: '' };

// A very small fetch() stub that understands /lessons/day-<N>.json
global.fetch = async (url) => {
  const u = String(url);
  if (u.startsWith('/lessons/day-') && u.endsWith('.json')) {
    const file = path.join(repoRoot, 'public', u.replace(/^[\/]/, ''));
    if (!fs.existsSync(file)) {
      return { ok: false, status: 404, json: async () => ({}) };
    }
    return {
      ok: true,
      status: 200,
      json: async () => JSON.parse(fs.readFileSync(file, 'utf8')),
    };
  }
  // Any other URL: simulate failure (we don't want network calls in this harness)
  return { ok: false, status: 500, json: async () => ({}) };
};

// AbortSignal.timeout used in loader; Node has AbortSignal, but timeout might not exist in older versions.
if (!global.AbortSignal || typeof global.AbortSignal.timeout !== 'function') {
  global.AbortSignal = {
    timeout: () => undefined,
  };
}

// -----------------------------
// Load the real loader script
// -----------------------------
const loaderJs = fs.readFileSync(path.join(repoRoot, 'public', 'js', 'kelly-lesson-loader.js'), 'utf8');
new Function(loaderJs)();

const Loader = global.window.KellyLessonLoader;
if (!Loader) {
  console.error('ERROR: window.KellyLessonLoader not found after evaluating script');
  process.exit(2);
}

const summarize = (result) => {
  const lesson = result?.lesson || {};
  const atoms = Array.isArray(result?.atoms) ? result.atoms : [];
  const phases = atoms.map((a) => ({
    phase: a.phase,
    options: Array.isArray(a?.content?.options) ? a.content.options.length : 0,
    optionLetters: Array.isArray(a?.content?.options) ? a.content.options.map(o => o.letter).join('') : '',
  }));
  return {
    source: result?._source || result?.source || '(unknown)',
    lesson: {
      day_number: lesson.day_number,
      topic: lesson.topic,
      category: lesson.category,
      emoji: lesson.emoji,
    },
    atoms_count: atoms.length,
    phases,
  };
};

(async () => {
  // 1) Day 1 seed happy path
  localStorage.clear();
  global.location.search = '';
  window.KELLY_CONFIG = { preferSeedLessons: true };

  const day1 = await Loader.getLesson(1, { archetype: 'The Scientist', age: 30, region: 'adult' });
  console.log('=== 1) Day 1 seed happy path: getLesson(1) ===');
  console.log(JSON.stringify(summarize(day1), null, 2));

  // 2) "Supabase returns skeleton/incomplete" -> directly prove ensureMvpLessonShape fills gaps
  const incomplete = {
    lesson: { id: 'supabase-pretend', day_number: 200, topic: 'INCOMPLETE TOPIC', universal_truth: '...' },
    atoms: [
      { phase: 'Hook', archetype: 'The Scientist', content: { script: 'hook only', options: [{ letter: 'A', text: 'A' }] } },
      { phase: 'Wisdom', archetype: 'The Scientist', content: { script: 'wisdom only' } },
    ],
    shards: [],
  };
  const filled = Loader.ensureMvpLessonShape(incomplete, { dayNum: 200, archetype: 'The Scientist', region: 'adult' });

  console.log('\n=== 2) ensureMvpLessonShape() on intentionally-incomplete payload ===');
  console.log(JSON.stringify(summarize(filled), null, 2));

  // 3) On-demand topic "volcanoes" via Grow track
  localStorage.clear();
  localStorage.setItem('kellyState', JSON.stringify({ track: 'grow', currentDay: 1 }));
  localStorage.setItem(Loader.ON_DEMAND_TOPIC_STORAGE_KEY, 'volcanoes');

  const onDemand = await Loader.getLesson(1, { archetype: 'The Explorer', age: 30, region: 'adult' });
  console.log('\n=== 3) On-demand "volcanoes" (Grow track) getLesson(1) ===');
  console.log(JSON.stringify(summarize(onDemand), null, 2));
})();
