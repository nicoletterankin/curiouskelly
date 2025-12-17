/**
 * Kelly Curriculum Browser
 * Displays 365-day curriculum organized by program year and month
 * 
 * Features:
 * - Program selector (Year 1, Year 2)
 * - Month-based navigation with expandable sections
 * - Daily topic display with learning objectives
 * - Search functionality
 * - Integration with lesson player
 */

'use strict';

const KellyCurriculumBrowser = (function() {
  
  // Curriculum metadata
  const PROGRAMS = [
    {
      id: 'year1',
      year: 1,
      name: 'Foundations of Knowledge',
      slug: 'year1-foundations',
      description: 'Daily lessons exploring the wonders of science, history, nature, and human achievement.',
      status: 'active',
      icon: '🌟'
    },
    {
      id: 'year2',
      year: 2,
      name: 'AI Fluency & Meta-Learning',
      slug: 'year2-ai-fluency',
      description: '365 days of learning how to learn in the age of AI—critical thinking, ethics, and human capabilities.',
      status: 'active',
      icon: '🧠'
    }
  ];

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

  // Year 2 theme colors and icons
  const YEAR2_THEMES = {
    January: { theme: 'Foundations', icon: '🏛️', color: '#3b82f6' },
    February: { theme: 'Questioning', icon: '❓', color: '#8b5cf6' },
    March: { theme: 'Verification', icon: '✓', color: '#10b981' },
    April: { theme: 'Memory & Learning', icon: '🧠', color: '#f59e0b' },
    May: { theme: 'Creativity & AI', icon: '🎨', color: '#ec4899' },
    June: { theme: 'Communication', icon: '💬', color: '#06b6d4' },
    July: { theme: 'Ethics & Responsibility', icon: '⚖️', color: '#6366f1' },
    August: { theme: 'Systems Thinking', icon: '🔗', color: '#14b8a6' },
    September: { theme: 'Human Capabilities', icon: '❤️', color: '#ef4444' },
    October: { theme: 'Privacy & Digital Citizenship', icon: '🔒', color: '#a855f7' },
    November: { theme: 'Future of Learning', icon: '🚀', color: '#22c55e' },
    December: { theme: 'Integration & Reflection', icon: '✨', color: '#f97316' }
  };

  // Cache for loaded curriculum data
  let curriculumCache = {};
  let currentProgram = 'year2';
  let searchQuery = '';
  let expandedMonths = new Set();

  /**
   * Initialize the curriculum browser
   */
  function init(containerId = 'curriculum-categories') {
    const container = document.getElementById(containerId);
    if (!container) {
      console.warn('[CurriculumBrowser] Container not found:', containerId);
      return;
    }

    // Render initial UI
    render(container);

    // Setup search listener
    const searchInput = document.getElementById('curriculum-search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value.toLowerCase().trim();
        render(container);
      });
    }

    // Load default program
    loadProgram(currentProgram, container);
  }

  /**
   * Render the curriculum browser UI
   */
  function render(container) {
    const html = `
      <div class="curriculum-browser">
        <!-- Program Selector -->
        <div class="program-selector">
          ${PROGRAMS.map(p => `
            <button class="program-btn ${currentProgram === p.id ? 'active' : ''}" 
                    data-program="${p.id}"
                    onclick="KellyCurriculumBrowser.selectProgram('${p.id}')">
              <span class="program-icon">${p.icon}</span>
              <span class="program-name">Year ${p.year}</span>
            </button>
          `).join('')}
        </div>

        <!-- Program Info -->
        <div class="program-info">
          ${renderProgramInfo()}
        </div>

        <!-- Months Grid -->
        <div class="curriculum-months" id="curriculum-months">
          ${renderMonths()}
        </div>
      </div>
    `;

    container.innerHTML = html;
  }

  /**
   * Render program info header
   */
  function renderProgramInfo() {
    const program = PROGRAMS.find(p => p.id === currentProgram);
    if (!program) return '';

    return `
      <div class="program-header">
        <div class="program-title">
          <span class="program-icon-large">${program.icon}</span>
          <div>
            <h2>${program.name}</h2>
            <p>${program.description}</p>
          </div>
        </div>
        <div class="program-stats">
          <div class="stat">
            <span class="stat-value">365</span>
            <span class="stat-label">Days</span>
          </div>
          <div class="stat">
            <span class="stat-value">12</span>
            <span class="stat-label">Themes</span>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Render all months
   */
  function renderMonths() {
    const cache = curriculumCache[currentProgram] || {};
    
    return MONTHS.map((month, index) => {
      const monthData = cache[month.name.toLowerCase()];
      const themeInfo = currentProgram === 'year2' ? YEAR2_THEMES[month.name] : { theme: 'Foundations', icon: '📚', color: '#3b82f6' };
      const isExpanded = expandedMonths.has(month.name);
      const lessonsLoaded = monthData && monthData.days && monthData.days.length > 0;
      
      // Filter by search query
      let filteredDays = [];
      if (lessonsLoaded && searchQuery) {
        filteredDays = monthData.days.filter(d => 
          d.title.toLowerCase().includes(searchQuery) ||
          d.learning_objective?.toLowerCase().includes(searchQuery)
        );
        if (filteredDays.length === 0 && !isExpanded) return ''; // Hide months with no matches
      } else if (lessonsLoaded) {
        filteredDays = monthData.days;
      }

      return `
        <div class="curriculum-month ${isExpanded ? 'expanded' : ''}" data-month="${month.name}">
          <div class="curriculum-month-header" onclick="KellyCurriculumBrowser.toggleMonth('${month.name}')">
            <div class="month-left">
              <div class="month-icon" style="background: ${themeInfo.color}">${themeInfo.icon}</div>
              <div class="month-info">
                <h3>${month.name}</h3>
                <span class="month-theme">${themeInfo.theme}</span>
                <span class="month-days">${month.days} days • Day ${month.startDay}-${month.startDay + month.days - 1}</span>
              </div>
            </div>
            <div class="month-right">
              <span class="month-arrow">${isExpanded ? '▼' : '▶'}</span>
            </div>
          </div>
          ${isExpanded ? `
            <div class="curriculum-month-content">
              ${lessonsLoaded ? renderDays(filteredDays, month) : `
                <div class="loading-days">
                  <div class="loading-spinner"></div>
                  <span>Loading ${month.name} curriculum...</span>
                </div>
              `}
            </div>
          ` : ''}
        </div>
      `;
    }).join('');
  }

  /**
   * Render days within a month
   */
  function renderDays(days, month) {
    if (!days || days.length === 0) {
      return '<div class="no-days">No lessons found</div>';
    }

    return `
      <div class="days-list">
        ${days.map(day => {
          // Get completion status from global state if available
          const isCompleted = window.state?.completedLessons?.includes(day.day) || false;
          const isToday = day.day === getTodayDayNumber();
          
          return `
            <div class="day-card ${isCompleted ? 'completed' : ''} ${isToday ? 'today' : ''}" 
                 onclick="KellyCurriculumBrowser.selectDay(${day.day})"
                 data-day="${day.day}">
              <div class="day-number">
                <span class="day-badge">${day.day}</span>
                ${isToday ? '<span class="today-badge">TODAY</span>' : ''}
              </div>
              <div class="day-content">
                <div class="day-title">${day.title}</div>
                <div class="day-objective">${truncate(day.learning_objective, 100)}</div>
              </div>
              <div class="day-status">
                ${isCompleted ? '✓' : '▶'}
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  }

  /**
   * Select a program (Year 1 or Year 2)
   */
  function selectProgram(programId) {
    if (currentProgram === programId) return;
    
    currentProgram = programId;
    expandedMonths.clear();
    
    const container = document.getElementById('curriculum-categories');
    if (container) {
      render(container);
      loadProgram(programId, container);
    }
  }

  /**
   * Toggle month expansion
   */
  function toggleMonth(monthName) {
    if (expandedMonths.has(monthName)) {
      expandedMonths.delete(monthName);
    } else {
      expandedMonths.add(monthName);
      // Load month data if not cached
      loadMonth(monthName);
    }
    
    const container = document.getElementById('curriculum-categories');
    if (container) render(container);
  }

  /**
   * Select a day to play
   */
  function selectDay(dayNumber) {
    console.log(`[CurriculumBrowser] Selected day ${dayNumber}`);
    
    // Close the panel
    if (typeof closePanel === 'function') {
      closePanel();
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
   * Load all curriculum for a program
   */
  async function loadProgram(programId, container) {
    if (!curriculumCache[programId]) {
      curriculumCache[programId] = {};
    }

    // Load all months in parallel
    const loadPromises = MONTHS.map(month => loadMonth(month.name.toLowerCase()));
    
    try {
      await Promise.all(loadPromises);
      render(container);
    } catch (error) {
      console.error('[CurriculumBrowser] Error loading program:', error);
    }
  }

  /**
   * Load a single month's curriculum
   */
  async function loadMonth(monthName) {
    const monthKey = monthName.toLowerCase();
    
    // Check cache
    if (curriculumCache[currentProgram]?.[monthKey]) {
      return curriculumCache[currentProgram][monthKey];
    }

    // Determine path based on program
    const basePath = currentProgram === 'year2' 
      ? '/data/curriculum/year2-ai-fluency'
      : '/data/curriculum/year1-foundations';
    
    const url = `${basePath}/${monthKey}_curriculum.json`;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!curriculumCache[currentProgram]) {
        curriculumCache[currentProgram] = {};
      }
      curriculumCache[currentProgram][monthKey] = data;
      
      return data;
    } catch (error) {
      console.warn(`[CurriculumBrowser] Could not load ${monthKey}:`, error.message);
      
      // Try fallback to lessons folder
      try {
        const fallbackUrl = currentProgram === 'year2'
          ? `/lessons/year2-ai-fluency/${monthKey}_curriculum.json`
          : `/lessons/year1-foundations/${monthKey}_curriculum.json`;
        
        const fallbackResponse = await fetch(fallbackUrl);
        if (fallbackResponse.ok) {
          const data = await fallbackResponse.json();
          if (!curriculumCache[currentProgram]) {
            curriculumCache[currentProgram] = {};
          }
          curriculumCache[currentProgram][monthKey] = data;
          return data;
        }
      } catch (fallbackError) {
        console.warn(`[CurriculumBrowser] Fallback also failed for ${monthKey}`);
      }
      
      return null;
    }
  }

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

  // Public API
  return {
    init,
    selectProgram,
    toggleMonth,
    selectDay,
    loadMonth,
    
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
