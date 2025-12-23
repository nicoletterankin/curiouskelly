/**
 * Kelly Curriculum Browser
 * Displays 365-day DUAL TRACK curriculum (Learn + Grow)
 * 
 * Features:
 * - Daily Duo view: Both tracks shown per day
 * - Month-based navigation with expandable sections
 * - Search across both tracks
 * - Integration with lesson player
 * 
 * @see docs/DUAL_TRACK_NAMING.md for naming conventions
 */

'use strict';

const KellyCurriculumBrowser = (function() {
  
  // Track definitions - LOCKED naming (see docs/DUAL_TRACK_NAMING.md)
  const TRACKS = {
    learn: {
      id: 'learn',
      name: 'Learn',
      fullName: 'Learn Track',
      slug: 'year1-foundations',
      description: 'What the world IS',
      icon: '/images/brand/icon-learn-track.svg',
      iconEmoji: '🌟', // Fallback for text contexts
      color: '#f59e0b'
    },
    grow: {
      id: 'grow',
      name: 'Grow',
      fullName: 'Grow Track',
      slug: 'year2-ai-fluency',
      description: 'How to LEARN',
      icon: '/images/brand/icon-grow-track.svg',
      iconEmoji: '🧠', // Fallback for text contexts
      color: '#8b5cf6'
    }
  };

  // Legacy support
  const PROGRAMS = [TRACKS.learn, TRACKS.grow];

  const MONTHS = [
    { name: 'January', days: 31, startDay: 1 },
    { name: 'February', days: 28, startDay: 32 },
    { name: 'March', days: 31, startDay: 60 },
    { name: 'April', days: 30, startDay: 91 },
    { name: 'May', days: 31, startDay: 121 },
    { name: 'June', days: 30, startDay: 152 },
    { name: 'July', days: 31, startDay: 182 },
    { name: 'August', days: 31, startDay: 213 },
    { name: 'September', days: 30, startDay: 244 },
    { name: 'October', days: 31, startDay: 274 },
    { name: 'November', days: 30, startDay: 305 },
    { name: 'December', days: 31, startDay: 335 }
  ];

  // Theme colors and icons for each track
  const TRACK_THEMES = {
    // 🧠 Grow Track (Meta-learning, AI Fluency)
    grow: {
      January: { theme: 'Foundations', icon: '🏛️', color: '#8b5cf6' },
      February: { theme: 'Questioning', icon: '❓', color: '#a855f7' },
      March: { theme: 'Verification', icon: '✓', color: '#7c3aed' },
      April: { theme: 'Memory & Learning', icon: '🧠', color: '#8b5cf6' },
      May: { theme: 'Creativity & AI', icon: '🎨', color: '#a855f7' },
      June: { theme: 'Communication', icon: '💬', color: '#7c3aed' },
      July: { theme: 'Ethics & Responsibility', icon: '⚖️', color: '#8b5cf6' },
      August: { theme: 'Systems Thinking', icon: '🔗', color: '#a855f7' },
      September: { theme: 'Human Capabilities', icon: '❤️', color: '#7c3aed' },
      October: { theme: 'Privacy & Digital Citizenship', icon: '🔒', color: '#8b5cf6' },
      November: { theme: 'Future of Learning', icon: '🚀', color: '#a855f7' },
      December: { theme: 'Integration & Reflection', icon: '✨', color: '#7c3aed' }
    },
    // 🌟 Learn Track (Knowledge, Discovery)
    learn: {
      January: { theme: 'Origins & Beginnings', icon: '🌅', color: '#f59e0b' },
      February: { theme: 'Life & Growth', icon: '🌱', color: '#eab308' },
      March: { theme: 'Forces & Motion', icon: '⚡', color: '#f59e0b' },
      April: { theme: 'Earth & Environment', icon: '🌍', color: '#eab308' },
      May: { theme: 'Innovation & Discovery', icon: '💡', color: '#f59e0b' },
      June: { theme: 'Art & Expression', icon: '🎨', color: '#eab308' },
      July: { theme: 'History & Civilization', icon: '🏛️', color: '#f59e0b' },
      August: { theme: 'Space & Universe', icon: '🚀', color: '#eab308' },
      September: { theme: 'Mind & Body', icon: '🧬', color: '#f59e0b' },
      October: { theme: 'Society & Culture', icon: '🌐', color: '#eab308' },
      November: { theme: 'Technology & Future', icon: '🔮', color: '#f59e0b' },
      December: { theme: 'Wonder & Wisdom', icon: '✨', color: '#eab308' }
    }
  };

  // Cache for loaded curriculum data (both tracks)
  let curriculumCache = { learn: {}, grow: {} };
  let viewMode = 'duo'; // 'duo' (both tracks) or 'single' (one track)
  let currentTrack = 'learn'; // For single-track view fallback
  let searchQuery = '';
  let expandedMonths = new Set();

  /**
   * Initialize the curriculum browser (Daily Duo mode)
   */
  function init(containerId = 'curriculum-categories') {
    const container = document.getElementById(containerId);
    if (!container) {
      console.warn('[CurriculumBrowser] Container not found:', containerId);
      return;
    }

    // Render initial UI
    render(container);

    // Load both tracks
    loadAllTracks(container);
  }

  /**
   * Render the curriculum browser UI - Daily Duo view
   */
  function render(container) {
    const html = `
      <div class="curriculum-browser curriculum-duo-mode">
        <!-- Header: Daily Duo branding -->
        <div class="duo-header">
          <div class="duo-title">
            <span class="duo-icon">📚</span>
            <div>
              <h2>365 Days of Learning</h2>
              <p class="duo-subtitle">Every day, two lessons: <span style="color: ${TRACKS.learn.color}"><img src="${TRACKS.learn.icon}" alt="${TRACKS.learn.name}" style="width: 1em; height: 1em; vertical-align: middle;" /> ${TRACKS.learn.name}</span> + <span style="color: ${TRACKS.grow.color}"><img src="${TRACKS.grow.icon}" alt="${TRACKS.grow.name}" style="width: 1em; height: 1em; vertical-align: middle;" /> ${TRACKS.grow.name}</span></p>
            </div>
          </div>
          <div class="duo-stats">
            <div class="stat"><span class="stat-value">730</span><span class="stat-label">Topics</span></div>
            <div class="stat"><span class="stat-value">365</span><span class="stat-label">Days</span></div>
          </div>
        </div>

        <!-- Track Legend -->
        <div class="track-legend">
          <div class="track-badge learn-badge" style="background: ${TRACKS.learn.color}20; border-color: ${TRACKS.learn.color}">
            <img src="${TRACKS.learn.icon}" alt="${TRACKS.learn.name}" style="width: 1.2em; height: 1.2em; vertical-align: middle;" /> <strong>${TRACKS.learn.name}</strong> — ${TRACKS.learn.description}
          </div>
          <div class="track-badge grow-badge" style="background: ${TRACKS.grow.color}20; border-color: ${TRACKS.grow.color}">
            <img src="${TRACKS.grow.icon}" alt="${TRACKS.grow.name}" style="width: 1.2em; height: 1.2em; vertical-align: middle;" /> <strong>${TRACKS.grow.name}</strong> — ${TRACKS.grow.description}
          </div>
        </div>

        <!-- Months Grid -->
        <div class="curriculum-months" id="curriculum-months">
          ${renderMonthsDuo()}
        </div>
      </div>
    `;

    container.innerHTML = html;
  }

  /**
   * Render months in Daily Duo mode (both tracks)
   */
  function renderMonthsDuo() {
    const learnCache = curriculumCache.learn || {};
    const growCache = curriculumCache.grow || {};
    
    return MONTHS.map((month, index) => {
      const learnData = learnCache[month.name.toLowerCase()];
      const growData = growCache[month.name.toLowerCase()];
      const isExpanded = expandedMonths.has(month.name);
      const dataLoaded = (learnData?.days?.length > 0) || (growData?.days?.length > 0);
      
      // Get theme info for display
      const learnTheme = TRACK_THEMES.learn[month.name] || { theme: 'Learning', icon: '📚' };
      const growTheme = TRACK_THEMES.grow[month.name] || { theme: 'Growing', icon: '🧠' };

      return `
        <div class="curriculum-month ${isExpanded ? 'expanded' : ''}" data-month="${month.name}">
          <div class="curriculum-month-header" onclick="KellyCurriculumBrowser.toggleMonth('${month.name}')">
            <div class="month-left">
              <div class="month-duo-icons">
                <span class="mini-icon learn" style="background: ${TRACKS.learn.color}"><img src="${TRACKS.learn.icon}" alt="${TRACKS.learn.name}" style="width: 16px; height: 16px;" /></span>
                <span class="mini-icon grow" style="background: ${TRACKS.grow.color}"><img src="${TRACKS.grow.icon}" alt="${TRACKS.grow.name}" style="width: 16px; height: 16px;" /></span>
              </div>
              <div class="month-info">
                <h3>${month.name}</h3>
                <span class="month-theme">${learnTheme.theme} + ${growTheme.theme}</span>
                <span class="month-days">${month.days} days × 2 tracks = ${month.days * 2} topics</span>
              </div>
            </div>
            <div class="month-right">
              <span class="month-arrow">${isExpanded ? '▼' : '▶'}</span>
            </div>
          </div>
          ${isExpanded ? `
            <div class="curriculum-month-content">
              ${dataLoaded ? renderDaysDuo(learnData?.days || [], growData?.days || [], month) : `
                <div class="loading-days">
                  <div class="loading-spinner"></div>
                  <span>Loading ${month.name}...</span>
                </div>
              `}
            </div>
          ` : ''}
        </div>
      `;
    }).join('');
  }

  /**
   * Get thumbnail URL for a lesson
   */
  function getThumbnailUrl(dayNumber, topic) {
    // 1. Try generic generated path (served by static files if they exist)
    // We don't check existence here (too expensive), but we use the convention
    // that might be handled by service worker or fallback
    
    // 2. Use Kelly Thumbnail Generator API if available (simulated here with path)
    // In a real app we might check if file exists, but here we'll default to a solid fallback
    // if we know specific ranges exist.
    
    // For now, use the hero image as a safe default for all lessons
    return '/images/kelly-hero-4k.webp';
  }

  /**
   * Render days with both Learn + Grow topics side by side
   */
  function renderDaysDuo(learnDays, growDays, month) {
    // Create a merged view by day number
    const dayCount = month.days;
    const startDay = month.startDay;
    
    let html = '<div class="days-list duo-days">';
    
    for (let i = 0; i < dayCount; i++) {
      const dayNum = startDay + i;
      const dateOfMonth = i + 1;
      
      const learnDay = learnDays.find(d => d.day === dayNum) || {};
      const growDay = growDays.find(d => d.day === dayNum) || {};
      
      const isToday = dayNum === getTodayDayNumber();
      const isCompleted = window.state?.completedLessons?.includes(dayNum) || false;
      
      const learnThumb = getThumbnailUrl(dayNum, learnDay.title);
      const growThumb = getThumbnailUrl(dayNum, growDay.title);
      
      html += `
        <div class="duo-day-card ${isToday ? 'today' : ''} ${isCompleted ? 'completed' : ''}" 
             data-day="${dayNum}">
          <div class="duo-day-header">
            <span class="duo-day-number">Day ${dayNum}</span>
            <span class="duo-day-date">${month.name} ${dateOfMonth}</span>
            ${isToday ? '<span class="today-badge">TODAY</span>' : ''}
          </div>
          <div class="duo-day-tracks">
            <div class="duo-track learn-track" onclick="KellyCurriculumBrowser.selectDay(${dayNum}, 'learn')">
              <div class="track-thumb" style="width: 48px; height: 48px; border-radius: 6px; overflow: hidden; margin-right: 12px; flex-shrink: 0;">
                <img src="${learnThumb}" alt="${learnDay.title || ''}" style="width: 100%; height: 100%; object-fit: cover;" loading="lazy" onerror="this.src='/images/kelly-hero-4k.webp'">
              </div>
              <div class="track-icon" style="background: ${TRACKS.learn.color}"><img src="${TRACKS.learn.icon}" alt="${TRACKS.learn.name}" style="width: 20px; height: 20px;" /></div>
              <div class="track-content">
                <div class="track-label">Learn</div>
                <div class="track-title">${learnDay.title || 'Loading...'}</div>
              </div>
              <div class="track-play">▶</div>
            </div>
            <div class="duo-track grow-track" onclick="KellyCurriculumBrowser.selectDay(${dayNum}, 'grow')">
              <div class="track-thumb" style="width: 48px; height: 48px; border-radius: 6px; overflow: hidden; margin-right: 12px; flex-shrink: 0;">
                <img src="${growThumb}" alt="${growDay.title || ''}" style="width: 100%; height: 100%; object-fit: cover;" loading="lazy" onerror="this.src='/images/kelly-hero-4k.webp'">
              </div>
              <div class="track-icon" style="background: ${TRACKS.grow.color}"><img src="${TRACKS.grow.icon}" alt="${TRACKS.grow.name}" style="width: 20px; height: 20px;" /></div>
              <div class="track-content">
                <div class="track-label">Grow</div>
                <div class="track-title">${growDay.title || 'Loading...'}</div>
              </div>
              <div class="track-play">▶</div>
            </div>
          </div>
        </div>
      `;
    }
    
    html += '</div>';
    return html;
  }

  // Legacy function for single-track view (kept for compatibility)
  function renderMonths() { return renderMonthsDuo(); }
  function renderDays(days, month) { return renderDaysDuo(days, [], month); }

  /**
   * Toggle month expansion (loads both tracks)
   */
  function toggleMonth(monthName) {
    if (expandedMonths.has(monthName)) {
      expandedMonths.delete(monthName);
    } else {
      expandedMonths.add(monthName);
      // Load both tracks for this month
      loadMonthDuo(monthName.toLowerCase());
    }
    
    const container = document.getElementById('curriculum-categories');
    if (container) render(container);
  }

  /**
   * Select a day to play
   */
  function selectDay(dayNumber, track = 'learn') {
    console.log(`[CurriculumBrowser] Selected day ${dayNumber}, track: ${track}`);
    
    // Close the journey mode if open
    if (typeof closeJourneyMode === 'function') {
      closeJourneyMode();
    }
    
    // Navigate to lesson
    if (typeof goToDay === 'function') {
      goToDay(dayNumber);
    } else if (typeof window.loadAndShowLesson === 'function') {
      window.loadAndShowLesson(dayNumber);
    } else {
      // Fallback: update URL and reload
      window.location.href = `/learn.html?day=${dayNumber}`;
    }
  }

  /**
   * Load both tracks for all months (initial load)
   */
  async function loadAllTracks(container) {
    const learnPromises = MONTHS.map(m => loadMonthForTrack('learn', m.name.toLowerCase()));
    const growPromises = MONTHS.map(m => loadMonthForTrack('grow', m.name.toLowerCase()));
    
    try {
      await Promise.all([...learnPromises, ...growPromises]);
      if (container) render(container);
    } catch (error) {
      console.error('[CurriculumBrowser] Error loading tracks:', error);
    }
  }

  /**
   * Load both tracks for a single month
   */
  async function loadMonthDuo(monthKey) {
    await Promise.all([
      loadMonthForTrack('learn', monthKey),
      loadMonthForTrack('grow', monthKey)
    ]);
    
    const container = document.getElementById('curriculum-categories');
    if (container) render(container);
  }

  /**
   * Load a single month's curriculum for a specific track
   */
  async function loadMonthForTrack(trackId, monthKey) {
    // Check cache
    if (curriculumCache[trackId]?.[monthKey]) {
      return curriculumCache[trackId][monthKey];
    }

    const pathMap = {
      'learn': 'year1-foundations',
      'grow': 'year2-ai-fluency'
    };
    const folderName = pathMap[trackId];
    const url = `/data/curriculum/${folderName}/${monthKey}_curriculum.json`;

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      
      if (!curriculumCache[trackId]) curriculumCache[trackId] = {};
      curriculumCache[trackId][monthKey] = data;
      
      return data;
    } catch (error) {
      console.warn(`[CurriculumBrowser] Could not load ${trackId}/${monthKey}:`, error.message);
      return null;
    }
  }

  // Legacy support
  function selectProgram(programId) { /* No-op in duo mode */ }
  function loadProgram(programId, container) { return loadAllTracks(container); }
  function loadMonth(monthName) { return loadMonthDuo(monthName.toLowerCase()); }

  /**
   * Helper: Get today's day number
   */
  function getTodayDayNumber() {
    if (typeof window.getTodayDayNumber === 'function') {
      return window.getTodayDayNumber();
    }
    
    const now = new Date();
    const start = new Date(now.getFullYear(), 0, 0);
    const diff = now - start;
    const oneDay = 1000 * 60 * 60 * 24;
    return Math.floor(diff / oneDay);
  }

  /**
   * Helper: Truncate text
   */
  function truncate(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  }

  /**
   * Search lessons across all months and tracks
   */
  function search(query) {
    searchQuery = (query || '').toLowerCase().trim();
    
    const container = document.getElementById('curriculum-months');
    if (!container) return;
    
    if (!searchQuery) {
      // Reset to normal view
      render(document.getElementById('curriculum-categories') || container.parentElement);
      loadAllTracks(container.parentElement);
      return;
    }
    
    // Search through cache
    const results = [];
    
    for (const [trackId, months] of Object.entries(curriculumCache)) {
      for (const [monthKey, monthData] of Object.entries(months)) {
        if (!monthData?.days) continue;
        
        for (const day of monthData.days) {
          const titleMatch = (day.title || '').toLowerCase().includes(searchQuery);
          const descMatch = (day.description || '').toLowerCase().includes(searchQuery);
          const tagMatch = (day.tags || []).some(t => t.toLowerCase().includes(searchQuery));
          
          if (titleMatch || descMatch || tagMatch) {
            results.push({
              ...day,
              track: trackId,
              trackInfo: TRACKS[trackId]
            });
          }
        }
      }
    }
    
    // Render search results
    renderSearchResults(container.parentElement, results);
  }
  
  /**
   * Render search results
   */
  function renderSearchResults(container, results) {
    if (results.length === 0) {
      container.innerHTML = `
        <div class="curriculum-browser">
          <div class="search-empty">
            <div class="search-empty-icon">🔍</div>
            <h3>No lessons found for "${searchQuery}"</h3>
            <p>Try different keywords or browse by month</p>
            <button class="clear-search-btn" onclick="KellyCurriculumBrowser.search('')">Clear Search</button>
          </div>
        </div>
      `;
      return;
    }
    
    container.innerHTML = `
      <div class="curriculum-browser">
        <div class="search-header">
          <h3>Found ${results.length} lesson${results.length !== 1 ? 's' : ''} for "${searchQuery}"</h3>
          <button class="clear-search-btn" onclick="KellyCurriculumBrowser.search('')">Clear</button>
        </div>
        <div class="search-results">
          ${results.slice(0, 50).map(day => `
            <div class="search-result-card" onclick="KellyCurriculumBrowser.selectDay(${day.day}, '${day.track}')">
              <div class="result-track" style="background: ${day.trackInfo?.color || '#666'}">
                ${day.trackInfo?.icon || '📚'} ${day.trackInfo?.name || 'Learn'}
              </div>
              <div class="result-info">
                <div class="result-day">Day ${day.day}</div>
                <div class="result-title">${day.title || 'Untitled'}</div>
                <div class="result-desc">${truncate(day.description || '', 80)}</div>
              </div>
              <div class="result-play">▶</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  // Public API
  return {
    init,
    selectProgram,
    toggleMonth,
    selectDay,
    loadMonth,
    search,
    
    // Expose for debugging
    getCache: () => curriculumCache,
    getCurrentProgram: () => currentProgram
  };
})();

// Auto-init when DOM ready (if container exists)
document.addEventListener('DOMContentLoaded', () => {
  // Don't auto-init - let the tab system call init when curriculum tab is selected
});

// Make globally accessible
window.KellyCurriculumBrowser = KellyCurriculumBrowser;
