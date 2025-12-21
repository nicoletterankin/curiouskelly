/**
 * Homepage Resource Explorer
 * - Lazy-loads an index of safe-to-list public files
 * - Simple search + filter
 * - Keyboard shortcut: Ctrl+K
 */
(() => {
  'use strict';

  const INDEX_URL = '/resources-index.json';
  const MAX_RESULTS = 24;

  const el = {
    details: document.getElementById('resources'),
    exploreBtn: document.getElementById('explore-btn'),
    search: document.getElementById('resources-search'),
    meta: document.getElementById('resources-meta'),
    results: document.getElementById('resources-results'),
    filterBtns: Array.from(document.querySelectorAll('.filter-btn'))
  };

  if (!el.details || !el.search || !el.meta || !el.results) return;

  /** @type {null | { version?: string; generatedAt?: string; items: any[] }} */
  let indexData = null;
  let activeFilter = 'all';
  let lastQuery = '';

  function safePath(p) {
    if (typeof p !== 'string') return null;
    if (!p.startsWith('/')) return null;
    if (p.includes('..')) return null;
    return p;
  }

  function normalize(s) {
    return String(s || '').toLowerCase().trim();
  }

  function getCategory(item) {
    const p = normalize(item.path || '');
    if (p.startsWith('/admin/')) return 'admin';
    if (p.startsWith('/data/') || p.startsWith('/lessons/') || p.endsWith('.json')) return 'data';
    if (p.startsWith('/js/') || p.startsWith('/css/') || p.endsWith('.js') || p.endsWith('.css')) return 'code';
    if (p.endsWith('.html') || p.endsWith('/')) return 'pages';
    return 'other';
  }

  function matchesFilter(item) {
    if (activeFilter === 'all') return true;
    return getCategory(item) === activeFilter;
  }

  function scoreItem(q, item) {
    // Simple heuristic scoring: title match > path match > tags match
    const title = normalize(item.title || '');
    const path = normalize(item.path || '');
    const tags = Array.isArray(item.tags) ? normalize(item.tags.join(' ')) : '';

    let score = 0;
    if (!q) return score;
    if (title.includes(q)) score += 5;
    if (path.includes(q)) score += 3;
    if (tags.includes(q)) score += 1;

    // bonus for startsWith
    if (title.startsWith(q)) score += 2;
    if (path.startsWith(q)) score += 1;
    return score;
  }

  function render(items, q) {
    el.results.innerHTML = '';

    if (!items.length) {
      el.meta.textContent = q
        ? `No matches for “${q}”.`
        : 'Type to search the index.';
      return;
    }

    const frag = document.createDocumentFragment();
    for (const item of items) {
      const href = safePath(item.path);
      if (!href) continue;

      const a = document.createElement('a');
      a.className = 'result';
      a.href = href;

      const title = document.createElement('div');
      title.className = 'result-title';
      title.textContent = item.title || href;

      const path = document.createElement('div');
      path.className = 'result-path';
      path.textContent = href;

      a.appendChild(title);
      a.appendChild(path);

      const tags = Array.isArray(item.tags) ? item.tags.slice(0, 4) : [];
      const cat = getCategory(item);
      const tagList = [cat, ...tags].filter(Boolean);
      if (tagList.length) {
        const tagsWrap = document.createElement('div');
        tagsWrap.className = 'result-tags';
        for (const t of tagList) {
          const span = document.createElement('span');
          span.className = 'tag';
          span.textContent = t;
          tagsWrap.appendChild(span);
        }
        a.appendChild(tagsWrap);
      }

      frag.appendChild(a);
    }

    el.results.appendChild(frag);
    const showing = Math.min(items.length, MAX_RESULTS);
    el.meta.textContent = `Showing ${showing}${items.length > MAX_RESULTS ? '+' : ''} results • filter: ${activeFilter}`;
  }

  function computeResults(q) {
    if (!indexData || !Array.isArray(indexData.items)) return [];
    const query = normalize(q);

    const filtered = indexData.items.filter(matchesFilter);
    if (!query) return filtered.slice(0, MAX_RESULTS);

    const scored = filtered
      .map((item) => ({ item, score: scoreItem(query, item) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score);

    return scored.slice(0, MAX_RESULTS).map((x) => x.item);
  }

  async function ensureIndexLoaded() {
    if (indexData) return indexData;
    el.meta.textContent = 'Loading index…';

    try {
      const res = await fetch(INDEX_URL, { cache: 'force-cache' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (!json || !Array.isArray(json.items)) throw new Error('Bad index format');
      indexData = json;
      return indexData;
    } catch (e) {
      el.meta.textContent = 'Could not load the resource index.';
      return null;
    }
  }

  async function openExplorer({ focusSearch = true } = {}) {
    el.details.open = true;
    await ensureIndexLoaded();
    // Initial render
    render(computeResults(el.search.value), normalize(el.search.value));
    if (focusSearch) {
      setTimeout(() => el.search.focus(), 0);
    }
  }

  function setFilter(next) {
    activeFilter = next;
    for (const btn of el.filterBtns) {
      btn.setAttribute('aria-pressed', btn.dataset.filter === next ? 'true' : 'false');
    }
    const q = el.search.value;
    render(computeResults(q), normalize(q));
  }

  // Events
  el.details.addEventListener('toggle', () => {
    if (el.details.open) {
      openExplorer({ focusSearch: false });
    }
  });

  if (el.exploreBtn) {
    el.exploreBtn.addEventListener('click', () => {
      openExplorer({ focusSearch: true });
    });
  }

  el.search.addEventListener('input', () => {
    const q = el.search.value;
    if (q === lastQuery) return;
    lastQuery = q;
    render(computeResults(q), normalize(q));
  });

  for (const btn of el.filterBtns) {
    btn.addEventListener('click', () => setFilter(btn.dataset.filter || 'all'));
  }

  // Keyboard shortcut
  document.addEventListener('keydown', (e) => {
    const isMac = /Mac|iPhone|iPad|iPod/.test(navigator.platform);
    const mod = isMac ? e.metaKey : e.ctrlKey;
    if (!mod) return;
    if (e.key.toLowerCase() !== 'k') return;

    e.preventDefault();
    openExplorer({ focusSearch: true });
  });
})();


