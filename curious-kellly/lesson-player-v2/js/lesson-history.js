/**
 * Lesson History Module
 * Handles spiral learning experience - tracking history, reflections, and milestones
 */

const LessonHistory = (function() {
  'use strict';

  // State
  let currentLessonDay = null;
  let lessonHistory = null;
  let reflectionData = null;

  /**
   * Get auth token from Supabase session
   */
  async function getAuthToken() {
    const { data: { session } } = await window.supabase.auth.getSession();
    return session?.access_token;
  }

  /**
   * Fetch lesson history for a specific day
   */
  async function fetchHistory(lessonDay) {
    const token = await getAuthToken();
    if (!token) return null;

    try {
      const response = await fetch(`/api/lesson-history?day=${lessonDay}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        console.warn('Failed to fetch lesson history');
        return null;
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching lesson history:', error);
      return null;
    }
  }

  /**
   * Fetch reflection data for a lesson
   */
  async function fetchReflection(lessonDay) {
    const token = await getAuthToken();
    if (!token) return null;

    try {
      const response = await fetch(`/api/reflection?day=${lessonDay}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) return null;
      return await response.json();
    } catch (error) {
      console.error('Error fetching reflection:', error);
      return null;
    }
  }

  /**
   * Record lesson completion
   */
  async function recordCompletion(lessonDay, answers, notes, timeSpentSeconds, layer) {
    const token = await getAuthToken();
    if (!token) return null;

    try {
      const response = await fetch('/api/lesson-complete', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          lessonDay,
          answers,
          notes,
          timeSpentSeconds,
          layer
        })
      });

      if (!response.ok) {
        console.warn('Failed to record completion');
        return null;
      }

      const result = await response.json();
      
      // Handle new milestones
      if (result.newMilestones && result.newMilestones.length > 0) {
        for (const milestone of result.newMilestones) {
          showMilestoneCelebration(milestone);
        }
      }

      // Show view number message
      if (result.viewNumber > 1) {
        showToast(`That's ${result.viewNumber} times you've learned this. ✨`);
      }

      return result;
    } catch (error) {
      console.error('Error recording completion:', error);
      return null;
    }
  }

  /**
   * Initialize history for a lesson
   */
  async function init(lessonDay) {
    currentLessonDay = lessonDay;
    
    // Fetch history
    lessonHistory = await fetchHistory(lessonDay);
    
    if (!lessonHistory) return;

    // Show returning learner banner if applicable
    if (lessonHistory.hasSeenBefore) {
      showReturningLearnerBanner(lessonHistory);
    }

    // Show birthday celebration if applicable
    if (lessonHistory.birthdayMessage) {
      showBirthdayCelebration(lessonHistory.birthdayMessage);
    }

    return lessonHistory;
  }

  /**
   * Show returning learner banner
   */
  function showReturningLearnerBanner(history) {
    // Remove existing banner if any
    const existingBanner = document.getElementById('returning-learner-banner');
    if (existingBanner) existingBanner.remove();

    const banner = document.createElement('div');
    banner.id = 'returning-learner-banner';
    banner.className = 'returning-learner-banner';
    
    const lastSeen = history.history[0] ? new Date(history.history[0].completedAt).toLocaleDateString('en-US', { month: 'long', year: 'numeric' }) : '';
    
    banner.innerHTML = `
      <div class="banner-content">
        <span class="view-count">You've learned this <strong>${history.viewCount} time${history.viewCount > 1 ? 's' : ''}</strong></span>
        ${lastSeen ? `<span class="last-seen">Last time: ${lastSeen}</span>` : ''}
        ${history.viewCount >= 2 ? `<button class="reflection-btn" onclick="LessonHistory.showReflection()">See how you've grown →</button>` : ''}
      </div>
      <button class="close-banner" onclick="this.parentElement.remove()">×</button>
    `;

    // Add styles if not already present
    if (!document.getElementById('lesson-history-styles')) {
      const styles = document.createElement('style');
      styles.id = 'lesson-history-styles';
      styles.textContent = `
        .returning-learner-banner {
          background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
          color: white;
          padding: 16px 24px;
          border-radius: 12px;
          margin: 16px 0;
          display: flex;
          justify-content: space-between;
          align-items: center;
          animation: slideDown 0.3s ease-out;
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .returning-learner-banner .banner-content {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          align-items: center;
        }
        .returning-learner-banner .view-count {
          font-size: 15px;
        }
        .returning-learner-banner .last-seen {
          font-size: 13px;
          opacity: 0.8;
        }
        .returning-learner-banner .reflection-btn {
          background: rgba(255,255,255,0.15);
          border: 1px solid rgba(255,255,255,0.3);
          color: white;
          padding: 8px 16px;
          border-radius: 8px;
          cursor: pointer;
          font-size: 13px;
          transition: background 0.2s;
        }
        .returning-learner-banner .reflection-btn:hover {
          background: rgba(255,255,255,0.25);
        }
        .returning-learner-banner .close-banner {
          background: none;
          border: none;
          color: white;
          font-size: 20px;
          cursor: pointer;
          opacity: 0.6;
          padding: 4px 8px;
        }
        .returning-learner-banner .close-banner:hover {
          opacity: 1;
        }

        .reflection-modal {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0,0,0,0.8);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 10000;
          animation: fadeIn 0.2s ease-out;
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .reflection-modal .modal-content {
          background: #18181b;
          border-radius: 16px;
          padding: 32px;
          max-width: 500px;
          width: 90%;
          max-height: 80vh;
          overflow-y: auto;
          color: #f4f4f5;
        }
        .reflection-modal h2 {
          margin: 0 0 24px;
          font-size: 24px;
          color: #f4f4f5;
        }
        .reflection-modal .timeline {
          margin-bottom: 24px;
        }
        .reflection-modal .year-entry {
          background: #27272a;
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 12px;
        }
        .reflection-modal .year-entry .year {
          font-weight: 600;
          color: #3b82f6;
        }
        .reflection-modal .year-entry .age {
          color: #a1a1aa;
          font-size: 14px;
          margin-left: 12px;
        }
        .reflection-modal .year-entry .answer {
          display: block;
          margin-top: 8px;
          color: #d4d4d8;
        }
        .reflection-modal .insight {
          background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
          border-radius: 8px;
          padding: 16px;
          font-style: italic;
          color: #e5e7eb;
          margin-bottom: 16px;
        }
        .reflection-modal .close-modal {
          background: #3b82f6;
          border: none;
          color: white;
          padding: 12px 24px;
          border-radius: 8px;
          cursor: pointer;
          font-size: 15px;
          width: 100%;
        }
        .reflection-modal .close-modal:hover {
          background: #2563eb;
        }

        .birthday-celebration {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0,0,0,0.9);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 10001;
          animation: fadeIn 0.3s ease-out;
        }
        .birthday-celebration .celebration-content {
          text-align: center;
          padding: 48px;
          max-width: 500px;
        }
        .birthday-celebration h1 {
          font-size: 48px;
          margin: 0 0 24px;
        }
        .birthday-celebration p {
          font-size: 20px;
          color: #d4d4d8;
          line-height: 1.6;
          margin: 0 0 32px;
        }
        .birthday-celebration button {
          background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
          border: none;
          color: white;
          padding: 16px 32px;
          border-radius: 50px;
          font-size: 18px;
          cursor: pointer;
          font-weight: 600;
        }

        .milestone-toast {
          position: fixed;
          bottom: 24px;
          right: 24px;
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          color: white;
          padding: 16px 24px;
          border-radius: 12px;
          z-index: 10000;
          animation: slideUp 0.3s ease-out;
          max-width: 300px;
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .milestone-toast .milestone-icon {
          font-size: 24px;
          margin-right: 12px;
        }
        .milestone-toast .milestone-text {
          font-size: 15px;
        }
      `;
      document.head.appendChild(styles);
    }

    // Insert at top of lesson container
    const lessonContainer = document.querySelector('.lesson-container, .learn-container, main, #app');
    if (lessonContainer) {
      lessonContainer.insertBefore(banner, lessonContainer.firstChild);
    }
  }

  /**
   * Show reflection modal
   */
  async function showReflection() {
    if (!currentLessonDay) return;

    // Fetch reflection data if not already loaded
    if (!reflectionData) {
      reflectionData = await fetchReflection(currentLessonDay);
    }

    if (!reflectionData || !reflectionData.canReflect) {
      showToast('Complete this lesson at least twice to see the reflection.');
      return;
    }

    const modal = document.createElement('div');
    modal.className = 'reflection-modal';
    modal.onclick = (e) => { if (e.target === modal) modal.remove(); };

    const timelineHtml = reflectionData.timeline.map(entry => `
      <div class="year-entry">
        <span class="year">${entry.year}</span>
        ${entry.age ? `<span class="age">Age ${entry.age}</span>` : ''}
        ${Object.entries(entry.answers || {}).map(([q, a]) => 
          `<span class="answer">${q.toUpperCase()}: "${a}"</span>`
        ).join('')}
      </div>
    `).join('');

    const insightsHtml = reflectionData.insights.map(insight => 
      `<p class="insight">${insight}</p>`
    ).join('');

    modal.innerHTML = `
      <div class="modal-content">
        <h2>Journey with This Lesson</h2>
        <div class="timeline">${timelineHtml}</div>
        ${insightsHtml}
        <button class="close-modal" onclick="this.closest('.reflection-modal').remove()">Continue Learning</button>
      </div>
    `;

    document.body.appendChild(modal);
  }

  /**
   * Show birthday celebration
   */
  function showBirthdayCelebration(message) {
    const celebration = document.createElement('div');
    celebration.className = 'birthday-celebration';

    celebration.innerHTML = `
      <div class="celebration-content">
        <h1>🎂 Happy Birthday!</h1>
        <p>${message}</p>
        <button onclick="this.closest('.birthday-celebration').remove()">Begin →</button>
      </div>
    `;

    document.body.appendChild(celebration);
  }

  /**
   * Show milestone celebration
   */
  function showMilestoneCelebration(milestone) {
    const icons = {
      'first_lesson': '🎉',
      'streak_7': '🔥',
      'streak_30': '💪',
      'streak_100': '⭐',
      'streak_365': '🏆',
      'streak_1000': '👑',
      'lessons_50': '📚',
      'lessons_100': '🎓',
      'lessons_200': '🌟',
      'lessons_365': '🎊',
      'year_complete_1': '🏅',
      'year_complete_5': '💎',
      'year_complete_10': '🌈'
    };

    const messages = {
      'first_lesson': 'First lesson complete! Welcome to curiosity.',
      'streak_7': '7 day streak! A week of learning.',
      'streak_30': '30 day streak! A month of curiosity.',
      'streak_100': '100 day streak! You\'re unstoppable.',
      'streak_365': '365 day streak! A full year. Legendary.',
      'streak_1000': '1000 day streak! You are extraordinary.',
      'lessons_50': '50 lessons learned!',
      'lessons_100': '100 lessons! You\'re truly curious.',
      'lessons_200': '200 lessons! Knowledge seeker.',
      'lessons_365': '365 lessons! You\'ve seen them all.',
      'year_complete_1': 'Year complete! All 365 lessons. Amazing.',
      'year_complete_5': '5 years complete! Half a decade of wonder.',
      'year_complete_10': 'Decade learner! 10 years. Extraordinary.'
    };

    const toast = document.createElement('div');
    toast.className = 'milestone-toast';
    toast.innerHTML = `
      <span class="milestone-icon">${icons[milestone.type] || '✨'}</span>
      <span class="milestone-text">${messages[milestone.type] || 'New milestone achieved!'}</span>
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = 'slideUp 0.3s ease-out reverse';
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  }

  /**
   * Show simple toast message
   */
  function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'milestone-toast';
    toast.style.background = 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)';
    toast.innerHTML = `<span class="milestone-text">${message}</span>`;

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = 'slideUp 0.3s ease-out reverse';
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  /**
   * Get recommended layer based on history
   */
  function getRecommendedLayer() {
    return lessonHistory?.recommendedLayer || 'foundation';
  }

  // Public API
  return {
    init,
    fetchHistory,
    fetchReflection,
    recordCompletion,
    showReflection,
    showBirthdayCelebration,
    showMilestoneCelebration,
    showToast,
    getRecommendedLayer,
    get history() { return lessonHistory; },
    get currentDay() { return currentLessonDay; }
  };
})();

// Make available globally
window.LessonHistory = LessonHistory;



