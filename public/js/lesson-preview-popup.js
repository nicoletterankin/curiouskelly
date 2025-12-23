/**
 * Lesson Preview Popup - Compact Completeness Display
 * Shows lesson completeness before playing, replaces full-screen audit
 */

(function() {
  'use strict';

  const LessonPreviewPopup = {
    /**
     * Calculate lesson completeness
     */
    calculateCompleteness(dayNumber) {
      const pack = window.CURIOUS_KELLY?.LOCAL_PACKS?.[dayNumber] || 
                   window.CURIOUS_KELLY?.LOCAL_PACKS?.[`day-${String(dayNumber).padStart(3, '0')}`] ||
                   window.CURIOUS_KELLY?.LOCAL_PACKS?.[String(dayNumber)];
      
      if (!pack) {
        return {
          completeness: 0,
          status: 'missing',
          checks: {
            learnBase: false,
            learnEnhanced: false,
            growBase: false,
            growEnhanced: false
          },
          stats: {
            phases: 0,
            videos: 0,
            visuals: 0,
            archetypes: 0
          }
        };
      }

      let score = 0;
      const checks = {
        learnBase: false,
        learnEnhanced: false,
        growBase: false,
        growEnhanced: false
      };
      
      const stats = {
        phases: 0,
        videos: 0,
        visuals: 0,
        archetypes: 0
      };

      // Learn base (40%): topic + 7 phases
      if (pack.lesson?.topic) {
        const atoms = pack.atoms || [];
        const phases = new Set(atoms.map(a => a.phase?.toLowerCase()));
        stats.phases = phases.size;
        
        if (phases.size >= 7) {
          checks.learnBase = true;
          score += 40;
        } else if (phases.size >= 4) {
          checks.learnBase = true;
          score += 20; // Partial
        }
      }

      // Learn enhanced (20%): videos, visuals, multiple archetypes
      const atoms = pack.atoms || [];
      const hasVideos = atoms.some(a => a.hd_video_url || a.video_url);
      const hasVisuals = atoms.some(a => a.visual_url);
      const archetypes = new Set(atoms.map(a => a.archetype).filter(Boolean));
      stats.archetypes = archetypes.size;
      stats.videos = atoms.filter(a => a.hd_video_url || a.video_url).length;
      stats.visuals = atoms.filter(a => a.visual_url).length;
      
      if (hasVideos || hasVisuals || archetypes.size > 1) {
        checks.learnEnhanced = true;
        score += 20;
      }

      // Grow base (30%): topic + objective
      if (pack.grow?.topic && pack.grow?.objective) {
        checks.growBase = true;
        score += 30;
      } else if (pack.grow?.topic) {
        checks.growBase = true;
        score += 15; // Partial
      }

      // Grow enhanced (10%): activity or full content
      if (pack.grow?.activity) {
        checks.growEnhanced = true;
        score += 10;
      }

      const status = score >= 80 ? 'production' :
                     score >= 60 ? 'complete' :
                     score >= 40 ? 'basic' : 'skeleton';

      return { completeness: Math.min(100, score), status, checks, stats };
    },

    /**
     * Show compact preview popup
     */
    async show(dayNumber) {
      // Close any existing popup
      this.close();

      // Try to load from LOCAL_PACKS first
      let pack = window.CURIOUS_KELLY?.LOCAL_PACKS?.[dayNumber] || 
                 window.CURIOUS_KELLY?.LOCAL_PACKS?.[`day-${String(dayNumber).padStart(3, '0')}`] ||
                 window.CURIOUS_KELLY?.LOCAL_PACKS?.[String(dayNumber)];

      // Fallback: Try to load from JSON file if not in LOCAL_PACKS
      if (!pack) {
        try {
          const response = await fetch(`/lessons/day-${dayNumber}.json`);
          if (response.ok) {
            const jsonData = await response.json();
            // Convert JSON format to pack format
            pack = {
              lesson: {
                topic: typeof jsonData.meta?.topic === 'object' ? jsonData.meta.topic.en : jsonData.meta?.topic,
                emoji: jsonData.meta?.emoji || '📚',
                category: jsonData.meta?.category || '',
                headline: typeof jsonData.headline === 'object' ? jsonData.headline.en : jsonData.headline,
                universal_truth: typeof jsonData.universal_truth === 'object' ? jsonData.universal_truth.en : jsonData.universal_truth
              },
              atoms: jsonData.phases ? Object.keys(jsonData.phases).map(phase => ({
                phase: phase,
                content: { script: jsonData.phases[phase]?.script || '' }
              })) : [],
              grow: jsonData.grow || null
            };
          }
        } catch (e) {
          console.warn('[LessonPreviewPopup] Could not load JSON fallback:', e);
        }
      }

      const completeness = this.calculateCompleteness(dayNumber, pack);
      
      // Get date
      let dateStr = `Day ${dayNumber}`;
      if (window.KellyTime?.dayNumberToDate) {
        const date = window.KellyTime.dayNumberToDate(dayNumber);
        dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }

      // Get topics (handle both LOCAL_PACKS and JSON formats)
      const learnTopic = pack?.lesson?.topic || 
                        (typeof pack?.meta?.topic === 'object' ? pack.meta.topic.en : pack?.meta?.topic) ||
                        'Loading...';
      const learnEmoji = pack?.lesson?.emoji || pack?.meta?.emoji || '📚';
      const growTopic = pack?.grow?.topic || 'Loading...';
      
      // Status badge colors
      const statusColors = {
        production: '#10b981',
        complete: '#3b82f6',
        basic: '#f59e0b',
        skeleton: '#6b7280',
        missing: '#ef4444'
      };
      const statusLabels = {
        production: 'Production Ready',
        complete: 'Complete',
        basic: 'Basic',
        skeleton: 'Skeleton',
        missing: 'Missing'
      };

      // Build popup HTML
      const popup = document.createElement('div');
      popup.className = 'lesson-preview-popup';
      popup.innerHTML = `
        <div class="preview-popup-backdrop" onclick="LessonPreviewPopup.close()"></div>
        <div class="preview-popup-card">
          <div class="preview-popup-header">
            <div class="preview-popup-title">
              <span class="preview-day-number">Day ${dayNumber}</span>
              <span class="preview-date">${dateStr}</span>
            </div>
            <button class="preview-popup-close" onclick="LessonPreviewPopup.close()">×</button>
          </div>
          
          <div class="preview-popup-content">
            <!-- Dual Track Topics -->
            <div class="preview-tracks">
              <div class="preview-track learn-track">
                <div class="track-label">
                  <span class="track-icon">📚</span>
                  <span>Learn Track</span>
                </div>
                <div class="track-topic">${learnEmoji} ${learnTopic}</div>
                <div class="track-status">
                  ${completeness.checks.learnBase ? '<span class="status-badge success">✓ Base</span>' : '<span class="status-badge missing">✗ Base</span>'}
                  ${completeness.checks.learnEnhanced ? '<span class="status-badge success">✓ Enhanced</span>' : '<span class="status-badge missing">✗ Enhanced</span>'}
                </div>
              </div>
              
              <div class="preview-track grow-track">
                <div class="track-label">
                  <span class="track-icon">🤖</span>
                  <span>Grow Track</span>
                </div>
                <div class="track-topic">${growTopic}</div>
                <div class="track-status">
                  ${completeness.checks.growBase ? '<span class="status-badge success">✓ Base</span>' : '<span class="status-badge missing">✗ Base</span>'}
                  ${completeness.checks.growEnhanced ? '<span class="status-badge success">✓ Enhanced</span>' : '<span class="status-badge missing">✗ Enhanced</span>'}
                </div>
              </div>
            </div>

            <!-- Completeness Indicator -->
            <div class="preview-completeness">
              <div class="completeness-header">
                <span class="completeness-label">Completeness</span>
                <span class="completeness-badge" style="background: ${statusColors[completeness.status]}">
                  ${statusLabels[completeness.status]}
                </span>
              </div>
              <div class="completeness-bar">
                <div class="completeness-fill" style="width: ${completeness.completeness}%; background: ${statusColors[completeness.status]}"></div>
                <span class="completeness-percent">${completeness.completeness}%</span>
              </div>
              
              <!-- Quick Stats -->
              <div class="completeness-stats">
                <div class="stat-item">
                  <span class="stat-label">Phases</span>
                  <span class="stat-value">${completeness.stats.phases}/7</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Videos</span>
                  <span class="stat-value">${completeness.stats.videos}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Visuals</span>
                  <span class="stat-value">${completeness.stats.visuals}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Archetypes</span>
                  <span class="stat-value">${completeness.stats.archetypes}</span>
                </div>
              </div>
            </div>

            <!-- Actions -->
            <div class="preview-actions">
              <a href="/learn.html?day=${dayNumber}&track=learn" class="preview-action-btn learn-btn">
                Start Learn Track →
              </a>
              <a href="/learn.html?day=${dayNumber}&track=grow" class="preview-action-btn grow-btn">
                Start Grow Track →
              </a>
              <button class="preview-action-btn secondary-btn" onclick="LessonPreviewPopup.showFullAudit(${dayNumber})">
                View Full Details
              </button>
            </div>
          </div>
        </div>
      `;

      document.body.appendChild(popup);
      this.injectStyles();

      // Close on Escape
      const escapeHandler = (e) => {
        if (e.key === 'Escape') {
          this.close();
          document.removeEventListener('keydown', escapeHandler);
        }
      };
      document.addEventListener('keydown', escapeHandler);
    },

    /**
     * Show full audit (right-side panel)
     */
    showFullAudit(dayNumber) {
      this.close();
      if (window.LessonAuditPanel) {
        window.LessonAuditPanel.show(dayNumber);
      } else if (window.LessonInspector) {
        window.LessonInspector.showInspector(dayNumber);
      } else if (window.LessonAudit) {
        window.LessonAudit.showInspector(dayNumber);
      }
    },

    /**
     * Close popup
     */
    close() {
      const popup = document.querySelector('.lesson-preview-popup');
      if (popup) popup.remove();
    },

    /**
     * Inject styles
     */
    injectStyles() {
      if (document.getElementById('lesson-preview-popup-styles')) return;

      const style = document.createElement('style');
      style.id = 'lesson-preview-popup-styles';
      style.textContent = `
        .lesson-preview-popup {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 10000;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
        }

        .preview-popup-backdrop {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(4px);
        }

        .preview-popup-card {
          position: relative;
          background: #1a1a1a;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          max-width: 600px;
          width: 100%;
          max-height: 90vh;
          overflow-y: auto;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
          z-index: 1;
        }

        .preview-popup-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 20px 24px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .preview-popup-title {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .preview-day-number {
          font-size: 20px;
          font-weight: 700;
          color: #fff;
        }

        .preview-date {
          font-size: 14px;
          color: rgba(255, 255, 255, 0.6);
        }

        .preview-popup-close {
          background: transparent;
          border: none;
          color: rgba(255, 255, 255, 0.6);
          font-size: 28px;
          cursor: pointer;
          padding: 0;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 6px;
          transition: all 0.15s;
        }

        .preview-popup-close:hover {
          background: rgba(255, 255, 255, 0.1);
          color: #fff;
        }

        .preview-popup-content {
          padding: 24px;
        }

        .preview-tracks {
          display: flex;
          flex-direction: column;
          gap: 16px;
          margin-bottom: 24px;
        }

        .preview-track {
          padding: 16px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .track-label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: rgba(255, 255, 255, 0.5);
          margin-bottom: 8px;
        }

        .track-icon {
          font-size: 16px;
        }

        .track-topic {
          font-size: 16px;
          font-weight: 600;
          color: #fff;
          margin-bottom: 8px;
        }

        .track-status {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }

        .status-badge {
          font-size: 11px;
          padding: 4px 8px;
          border-radius: 6px;
          font-weight: 600;
        }

        .status-badge.success {
          background: rgba(16, 185, 129, 0.2);
          color: #10b981;
        }

        .status-badge.missing {
          background: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }

        .preview-completeness {
          margin-bottom: 24px;
        }

        .completeness-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 12px;
        }

        .completeness-label {
          font-size: 14px;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.7);
        }

        .completeness-badge {
          font-size: 12px;
          padding: 4px 12px;
          border-radius: 12px;
          font-weight: 600;
          color: #fff;
        }

        .completeness-bar {
          position: relative;
          height: 32px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          overflow: hidden;
          margin-bottom: 12px;
        }

        .completeness-fill {
          height: 100%;
          transition: width 0.3s ease;
          border-radius: 8px;
        }

        .completeness-percent {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          font-size: 14px;
          font-weight: 700;
          color: #fff;
        }

        .completeness-stats {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 12px;
        }

        .stat-item {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .stat-label {
          font-size: 11px;
          color: rgba(255, 255, 255, 0.5);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .stat-value {
          font-size: 18px;
          font-weight: 700;
          color: #fff;
        }

        .preview-actions {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .preview-action-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 14px 20px;
          border-radius: 10px;
          font-size: 15px;
          font-weight: 600;
          text-decoration: none;
          border: none;
          cursor: pointer;
          transition: all 0.15s;
        }

        .preview-action-btn.learn-btn {
          background: #f59e0b;
          color: #fff;
        }

        .preview-action-btn.learn-btn:hover {
          background: #d97706;
          transform: translateY(-1px);
        }

        .preview-action-btn.grow-btn {
          background: #8b5cf6;
          color: #fff;
        }

        .preview-action-btn.grow-btn:hover {
          background: #7c3aed;
          transform: translateY(-1px);
        }

        .preview-action-btn.secondary-btn {
          background: transparent;
          color: rgba(255, 255, 255, 0.7);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .preview-action-btn.secondary-btn:hover {
          background: rgba(255, 255, 255, 0.05);
          color: #fff;
        }

        @media (max-width: 600px) {
          .preview-popup-card {
            max-width: 100%;
            border-radius: 12px;
          }

          .completeness-stats {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `;

      document.head.appendChild(style);
    }
  };

  // Export
  window.LessonPreviewPopup = LessonPreviewPopup;
})();

