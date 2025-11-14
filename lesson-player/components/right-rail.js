/**
 * Right-Rail UI Components
 * 
 * Implements the right-side navigation icons:
 * - Live: Shows live state countdown and current phase
 * - Find: Search within lesson content
 * - Settings/Controls: Playback speed, language, accessibility
 * - Calendar: y/y/t format showing progress
 * 
 * As per CLAUDE.md requirements.
 */

class RightRailUI {
  constructor(containerElement) {
    this.container = containerElement;
    this.activePanel = null;
    this.lessonData = null;
    this.currentPhase = 'welcome';
    this.currentTime = 0;
    this.phaseOrder = ['welcome', 'main', 'wisdom'];

    this.init();
  }

  /**
   * Initialize right-rail UI
   */
  init() {
    this.render();
    this.setupEventListeners();
  }

  /**
   * Render right-rail icons and panels
   */
  render() {
    this.container.innerHTML = `
      <div class="right-rail">
        <div class="right-rail-icons">
          <button class="right-rail-icon" data-panel="live" title="Live State">
            <svg width="24" height="24" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/>
              <circle cx="12" cy="12" r="3" fill="currentColor"/>
            </svg>
            <span class="icon-label">Live</span>
          </button>

          <button class="right-rail-icon" data-panel="search" title="Find">
            <svg width="24" height="24" viewBox="0 0 24 24">
              <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2"/>
              <path d="M16 16 L21 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <span class="icon-label">Find</span>
          </button>

          <button class="right-rail-icon" data-panel="settings" title="Settings">
            <svg width="24" height="24" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="2"/>
              <path d="M12 1 L12 5 M12 19 L12 23 M4.22 4.22 L7.05 7.05 M16.95 16.95 L19.78 19.78 M1 12 L5 12 M19 12 L23 12 M4.22 19.78 L7.05 16.95 M16.95 7.05 L19.78 4.22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <span class="icon-label">Settings</span>
          </button>

          <button class="right-rail-icon" data-panel="calendar" title="Calendar">
            <svg width="24" height="24" viewBox="0 0 24 24">
              <rect x="3" y="4" width="18" height="18" rx="2" fill="none" stroke="currentColor" stroke-width="2"/>
              <path d="M3 10 L21 10 M8 2 L8 6 M16 2 L16 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <span class="icon-label">Calendar</span>
          </button>
        </div>

        <div class="right-rail-panels">
          <div class="right-rail-panel" id="panel-live">
            <h3>Live State</h3>
            <div class="live-content">
              <div class="phase-indicator">
                <span class="phase-label">Current Phase:</span>
                <span class="phase-name" id="current-phase-name">Welcome</span>
              </div>
              <div class="time-remaining">
                <span class="time-label">Time Remaining:</span>
                <span class="time-value" id="time-remaining">--:--</span>
              </div>
              <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
              </div>
            </div>
          </div>

          <div class="right-rail-panel" id="panel-search">
            <h3>Find</h3>
            <div class="search-content">
              <input type="text" id="search-input" class="search-input" placeholder="Search lesson content...">
              <div class="search-results" id="search-results"></div>
            </div>
          </div>

          <div class="right-rail-panel" id="panel-settings">
            <h3>Settings & Controls</h3>
            <div class="settings-content">
              <div class="setting-group">
                <label for="playback-speed">Playback Speed</label>
                <select id="playback-speed" class="setting-control">
                  <option value="0.75">0.75x</option>
                  <option value="1.0" selected>1.0x</option>
                  <option value="1.25">1.25x</option>
                  <option value="1.5">1.5x</option>
                </select>
              </div>

              <div class="setting-group">
                <label for="language-select">Language</label>
                <select id="language-select" class="setting-control">
                  <option value="en" selected>English</option>
                  <option value="es">Español</option>
                  <option value="fr">Français</option>
                </select>
              </div>

              <div class="setting-group">
                <label>
                  <input type="checkbox" id="show-subtitles" checked>
                  Show Subtitles
                </label>
              </div>

              <div class="setting-group">
                <label>
                  <input type="checkbox" id="high-contrast">
                  High Contrast Mode
                </label>
              </div>

              <div class="setting-group">
                <label>
                  <input type="checkbox" id="reduce-motion">
                  Reduce Motion
                </label>
              </div>
            </div>
          </div>

          <div class="right-rail-panel" id="panel-calendar">
            <h3>Progress Calendar</h3>
            <div class="calendar-content">
              <div class="calendar-format">
                <span class="calendar-label">Format: y/y/t</span>
                <div class="calendar-display">
                  <div class="calendar-item">
                    <span class="calendar-value" id="yesterday-status">✓</span>
                    <span class="calendar-sublabel">Yesterday</span>
                  </div>
                  <div class="calendar-item">
                    <span class="calendar-value" id="today-status">●</span>
                    <span class="calendar-sublabel">Today</span>
                  </div>
                  <div class="calendar-item">
                    <span class="calendar-value" id="tomorrow-status">○</span>
                    <span class="calendar-sublabel">Tomorrow</span>
                  </div>
                </div>
              </div>

              <div class="streak-info">
                <span class="streak-label">Current Streak:</span>
                <span class="streak-value" id="streak-count">7 days</span>
              </div>

              <div class="progress-summary">
                <p><strong>Total Lessons:</strong> <span id="total-lessons">4</span></p>
                <p><strong>Completed:</strong> <span id="completed-lessons">2</span></p>
                <p><strong>Progress:</strong> <span id="overall-progress">50%</span></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    // Icon click handlers
    const icons = this.container.querySelectorAll('.right-rail-icon');
    icons.forEach(icon => {
      icon.addEventListener('click', (e) => {
        const panelName = icon.dataset.panel;
        this.togglePanel(panelName);
      });
    });

    // Search input
    const searchInput = this.container.querySelector('#search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.performSearch(e.target.value);
      });
    }

    // Playback speed
    const speedSelect = this.container.querySelector('#playback-speed');
    if (speedSelect) {
      speedSelect.addEventListener('change', (e) => {
        this.setPlaybackSpeed(parseFloat(e.target.value));
      });
    }

    // Language select
    const langSelect = this.container.querySelector('#language-select');
    if (langSelect) {
      langSelect.addEventListener('change', (e) => {
        this.setLanguage(e.target.value);
      });
    }

    // Settings checkboxes
    const showSubtitles = this.container.querySelector('#show-subtitles');
    if (showSubtitles) {
      showSubtitles.addEventListener('change', (e) => {
        this.toggleSubtitles(e.target.checked);
      });
    }

    const highContrast = this.container.querySelector('#high-contrast');
    if (highContrast) {
      highContrast.addEventListener('change', (e) => {
        this.toggleHighContrast(e.target.checked);
      });
    }

    const reduceMotion = this.container.querySelector('#reduce-motion');
    if (reduceMotion) {
      reduceMotion.addEventListener('change', (e) => {
        this.toggleReduceMotion(e.target.checked);
      });
    }
  }

  /**
   * Toggle panel visibility
   */
  togglePanel(panelName) {
    const panel = this.container.querySelector(`#panel-${panelName}`);
    const icon = this.container.querySelector(`.right-rail-icon[data-panel="${panelName}"]`);

    if (this.activePanel === panelName) {
      // Close current panel
      panel.classList.remove('active');
      icon.classList.remove('active');
      this.activePanel = null;
    } else {
      // Close other panels
      this.container.querySelectorAll('.right-rail-panel.active').forEach(p => {
        p.classList.remove('active');
      });
      this.container.querySelectorAll('.right-rail-icon.active').forEach(i => {
        i.classList.remove('active');
      });

      // Open new panel
      panel.classList.add('active');
      icon.classList.add('active');
      this.activePanel = panelName;
    }
  }

  /**
   * Update live state display
   */
  updateLiveState(phase, currentTime, totalTime) {
    this.currentPhase = phase;
    this.currentTime = currentTime;

    const phaseNameEl = this.container.querySelector('#current-phase-name');
    const timeRemainingEl = this.container.querySelector('#time-remaining');
    const progressFillEl = this.container.querySelector('#progress-fill');

    if (phaseNameEl) {
      phaseNameEl.textContent = phase.charAt(0).toUpperCase() + phase.slice(1);
    }

    if (timeRemainingEl && totalTime) {
      const remaining = Math.max(0, totalTime - currentTime);
      const minutes = Math.floor(remaining / 60);
      const seconds = Math.floor(remaining % 60);
      timeRemainingEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    if (progressFillEl && totalTime) {
      const progress = (currentTime / totalTime) * 100;
      progressFillEl.style.width = `${Math.min(100, progress)}%`;
    }
  }

  /**
   * Perform search in lesson content
   */
  performSearch(query) {
    const resultsEl = this.container.querySelector('#search-results');
    if (!resultsEl || !this.lessonData) return;

    if (!query || query.length < 2) {
      resultsEl.innerHTML = '<p class="search-hint">Enter at least 2 characters to search</p>';
      return;
    }

    // Search in lesson content
    const results = [];
    const searchLower = query.toLowerCase();

    // Search in welcome, main content, wisdom
    ['welcome', 'mainContent', 'wisdomMoment'].forEach(field => {
      const content = this.lessonData?.language?.[field];
      if (content && content.toLowerCase().includes(searchLower)) {
        results.push({
          phase: field,
          text: content
        });
      }
    });

    if (results.length === 0) {
      resultsEl.innerHTML = '<p class="no-results">No results found</p>';
    } else {
      resultsEl.innerHTML = results.map(result => `
        <div class="search-result" data-phase="${result.phase}">
          <strong>${result.phase}</strong>
          <p>${this.highlightQuery(result.text, query)}</p>
        </div>
      `).join('');
    }
  }

  /**
   * Highlight search query in text
   */
  highlightQuery(text, query) {
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  }

  /**
   * Set playback speed
   */
  setPlaybackSpeed(speed) {
    // Dispatch event for lesson player to handle
    window.dispatchEvent(new CustomEvent('playbackSpeedChanged', { detail: { speed } }));
  }

  /**
   * Set language
   */
  setLanguage(language) {
    window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language } }));
  }

  /**
   * Toggle subtitles
   */
  toggleSubtitles(enabled) {
    window.dispatchEvent(new CustomEvent('subtitlesToggled', { detail: { enabled } }));
  }

  /**
   * Toggle high contrast mode
   */
  toggleHighContrast(enabled) {
    if (enabled) {
      document.body.classList.add('high-contrast');
    } else {
      document.body.classList.remove('high-contrast');
    }
  }

  /**
   * Toggle reduced motion
   */
  toggleReduceMotion(enabled) {
    if (enabled) {
      document.body.classList.add('reduce-motion');
    } else {
      document.body.classList.remove('reduce-motion');
    }
  }

  /**
   * Set lesson data for searching
   */
  setLessonData(lessonData) {
    this.lessonData = lessonData;
  }

  /**
   * Update calendar display
   */
  updateCalendar(yesterdayComplete, todayInProgress, streak) {
    const yesterdayEl = this.container.querySelector('#yesterday-status');
    const todayEl = this.container.querySelector('#today-status');
    const tomorrowEl = this.container.querySelector('#tomorrow-status');
    const streakEl = this.container.querySelector('#streak-count');

    if (yesterdayEl) {
      yesterdayEl.textContent = yesterdayComplete ? '✓' : '○';
    }

    if (todayEl) {
      todayEl.textContent = todayInProgress ? '●' : '○';
    }

    if (tomorrowEl) {
      tomorrowEl.textContent = '○'; // Tomorrow always pending
    }

    if (streakEl) {
      streakEl.textContent = `${streak} day${streak !== 1 ? 's' : ''}`;
    }
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = RightRailUI;
}



