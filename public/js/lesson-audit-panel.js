/**
 * Lesson Audit Panel - Right-Side Panel with Dual Views
 * Learner-First View: What learners will experience
 * Educator View: Full technical blueprint
 */

(function() {
  'use strict';

  const LessonAuditPanel = {
    currentDay: null,
    currentView: 'learner', // 'learner' | 'educator'
    auditData: null,
    isLoading: false,

    /**
     * Show the audit panel for a specific day
     */
    async show(dayNumber) {
      this.currentDay = dayNumber;
      this.isLoading = true;
      
      // Create panel if it doesn't exist
      if (!document.getElementById('audit-panel')) {
        this.createPanel();
      }
      
      const panel = document.getElementById('audit-panel');
      panel.classList.add('open');
      document.body.classList.add('audit-panel-open');
      document.body.style.overflow = 'hidden'; // Prevent background scroll
      
      // Load audit data
      await this.loadAudit(dayNumber);
      
      // Render based on current view
      if (this.currentView === 'learner') {
        this.renderLearnerView();
      } else {
        this.renderEducatorView();
      }
      
      this.isLoading = false;
    },

    /**
     * Close the panel
     */
    close() {
      const panel = document.getElementById('audit-panel');
      if (panel) {
        panel.classList.remove('open');
        document.body.classList.remove('audit-panel-open');
        document.body.style.overflow = ''; // Restore scroll
      }
    },

    /**
     * Toggle between learner and educator views
     */
    toggleView() {
      this.currentView = this.currentView === 'learner' ? 'educator' : 'learner';
      
      // Update button state
      const learnerBtn = document.getElementById('audit-view-learner');
      const educatorBtn = document.getElementById('audit-view-educator');
      learnerBtn?.classList.toggle('active', this.currentView === 'learner');
      educatorBtn?.classList.toggle('active', this.currentView === 'educator');
      
      // Re-render
      if (this.currentView === 'learner') {
        this.renderLearnerView();
      } else {
        this.renderEducatorView();
      }
    },

    /**
     * Load audit data for a day
     * Uses KellyUnifiedLessonService for data coherency
     */
    async loadAudit(dayNumber) {
      // Use unified service as primary source
      if (window.KellyUnifiedLessonService) {
        try {
          const preview = await window.KellyUnifiedLessonService.getPreview(dayNumber, { track: 'both' });
          
          this.auditData = {
            dayNumber: preview.dayNumber,
            date: preview.date,
            tracks: {
              learn: {
                topic: preview.learn.topic || '',
                emoji: preview.learn.emoji || '📚',
                category: preview.learn.category || '',
                headline: preview.learn.headline || '',
                universalTruth: preview.learn.universalTruth || '',
                atoms: preview.learn.atoms || [],
                phases: preview.learn.phases || {},
                visuals: preview.learn.visuals || [],
                videos: preview.learn.videos || [],
                completeness: preview.completeness,
                status: preview.status,
                checks: preview.checks,
                stats: preview.stats
              },
              grow: preview.grow ? {
                topic: preview.grow.topic || '',
                objective: preview.grow.objective || '',
                activity: preview.grow.activity || '',
                emoji: preview.grow.emoji || '🤖'
              } : null
            },
            assets: {
              videos: preview.learn.videos || [],
              visuals: preview.learn.visuals || [],
              audio: []
            },
            sources: {
              unified: true
            },
            variants: { languages: new Set(), archetypes: new Set(), ageBuckets: new Set() },
            errors: [],
            warnings: []
          };
          
          return;
        } catch (e) {
          console.warn('[AuditPanel] Unified service failed, falling back:', e);
        }
      }
      
      // Fallback: Use existing LessonInspector if available
      if (window.LessonInspector && typeof window.LessonInspector.getFullAudit === 'function') {
        this.auditData = await window.LessonInspector.getFullAudit(dayNumber);
        
        // Enhance audit data with completeness calculation
        const completeness = window.LessonPreviewPopup?.calculateCompleteness(dayNumber);
        if (completeness) {
          this.auditData.tracks = this.auditData.tracks || {};
          this.auditData.tracks.learn = {
            ...this.auditData.tracks.learn,
            completeness: completeness.completeness,
            status: completeness.status,
            checks: completeness.checks,
            stats: completeness.stats
          };
        }
        
        // Extract Grow track from audit assets
        if (this.auditData.assets?.growTrack || this.auditData.assets?.growTrackSupabase) {
          const growTrack = this.auditData.assets.growTrack || this.auditData.assets.growTrackSupabase;
          this.auditData.tracks.grow = {
            topic: typeof growTrack.topic === 'object' ? growTrack.topic.en : growTrack.topic || 
                   (typeof growTrack.title === 'object' ? growTrack.title.en : growTrack.title),
            objective: typeof growTrack.learning_objective === 'object' ? growTrack.learning_objective.en : growTrack.learning_objective,
            activity: typeof growTrack.activity === 'object' ? growTrack.activity.en : growTrack.activity,
            emoji: growTrack.emoji || '🤖'
          };
        }
      } else {
        // Last resort fallback
        const completeness = window.LessonPreviewPopup?.calculateCompleteness(dayNumber);
        this.auditData = {
          dayNumber,
          tracks: {
            learn: completeness || { completeness: 0, status: 'missing', checks: {}, stats: {} },
            grow: {}
          },
          assets: {},
          sources: {},
          variants: { languages: new Set(), archetypes: new Set(), ageBuckets: new Set() },
          errors: [],
          warnings: []
        };
      }
      
      // Enhance with Grow track data if not already loaded
      if (!this.auditData.tracks?.grow?.topic) {
        await this.loadGrowTrackData(dayNumber);
      }
    },

    /**
     * Load Grow track data
     */
    async loadGrowTrackData(dayNumber) {
      try {
        // Try to load from LOCAL_PACKS
        const pack = window.CURIOUS_KELLY?.LOCAL_PACKS?.[dayNumber] ||
                     window.CURIOUS_KELLY?.LOCAL_PACKS?.[`day-${String(dayNumber).padStart(3, '0')}`] ||
                     window.CURIOUS_KELLY?.LOCAL_PACKS?.[String(dayNumber)];
        
        if (pack?.grow) {
          this.auditData.tracks.grow = {
            topic: pack.grow.topic,
            objective: pack.grow.objective,
            activity: pack.grow.activity,
            emoji: pack.grow.emoji || '🤖'
          };
        } else {
          // Try JSON file
          const response = await fetch(`/lessons/day-${dayNumber}.json`);
          if (response.ok) {
            const jsonData = await response.json();
            if (jsonData.growTrack || jsonData.grow) {
              const grow = jsonData.growTrack || jsonData.grow;
              this.auditData.tracks.grow = {
                topic: typeof grow.topic === 'object' ? grow.topic.en : grow.topic,
                objective: typeof grow.learning_objective === 'object' ? grow.learning_objective.en : grow.learning_objective,
                activity: typeof grow.activity === 'object' ? grow.activity.en : grow.activity,
                emoji: grow.emoji || '🤖'
              };
            }
          }
        }
      } catch (e) {
        // Silent fail - Grow track is optional
      }
    },

    /**
     * Create the panel HTML structure
     */
    createPanel() {
      const panel = document.createElement('div');
      panel.id = 'audit-panel';
      panel.className = 'audit-panel';
      panel.innerHTML = `
        <div class="audit-panel-overlay" onclick="LessonAuditPanel.close()"></div>
        <div class="audit-panel-content">
          <div class="audit-panel-header">
            <div class="audit-panel-title">
              <span id="audit-day-label">Day 1</span>
              <span class="audit-panel-subtitle" id="audit-date-label"></span>
            </div>
            <button class="audit-panel-close" onclick="LessonAuditPanel.close()">×</button>
          </div>
          
          <div class="audit-view-toggle">
            <button id="audit-view-learner" class="audit-view-btn active" onclick="LessonAuditPanel.toggleView()">
              👤 Learner View
            </button>
            <button id="audit-view-educator" class="audit-view-btn" onclick="LessonAuditPanel.toggleView()">
              🔧 Educator View
            </button>
          </div>
          
          <div class="audit-panel-body" id="audit-panel-body">
            <div class="audit-loading">Loading lesson details...</div>
          </div>
        </div>
      `;
      
      document.body.appendChild(panel);
      this.injectStyles();
    },

    /**
     * Render learner-first view
     */
    renderLearnerView() {
      if (!this.auditData) return;
      
      const body = document.getElementById('audit-panel-body');
      if (!body) return;
      
      const day = this.auditData.dayNumber || this.currentDay;
      const learn = this.auditData.tracks?.learn || {};
      const grow = this.auditData.tracks?.grow || {};
      const completeness = learn.completeness || 0;
      const status = learn.status || 'missing';
      
      // Get date
      let dateStr = '';
      if (window.KellyTime?.dayNumberToDate) {
        const date = window.KellyTime.dayNumberToDate(day);
        dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      }
      
      // Status colors
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
      
      // Get lesson data from auditData (loaded via unified service)
      const learnTopic = learn.topic || 'Lesson not available';
      const learnEmoji = learn.emoji || '📚';
      const learnHeadline = learn.headline || '';
      const learnUniversalTruth = learn.universalTruth || '';
      const growTopic = grow?.topic || 'Not available';
      const growEmoji = grow?.emoji || '🤖';
      
      // Get phases, visuals, videos from unified data
      const phases = learn.phases || {};
      const visuals = learn.visuals || [];
      const videos = learn.videos || [];
      const atoms = learn.atoms || [];
      
      // Calculate asset counts
      const videoCount = videos.length;
      const visualCount = visuals.length;
      const phaseCount = Object.keys(phases).length;
      const archetypeCount = new Set(atoms.map(a => a.archetype).filter(Boolean)).size;
      
      body.innerHTML = `
        <div class="audit-learner-view">
          <!-- Completeness Badge -->
          <div class="learner-completeness">
            <div class="completeness-gauge-large">
              <div class="gauge-fill-large" style="width: ${completeness}%; background: ${statusColors[status]}"></div>
              <div class="gauge-label-large">${completeness}% ${statusLabels[status]}</div>
            </div>
          </div>
          
          <!-- Quick Preview -->
          <div class="learner-preview-section">
            <h3 class="section-title">📚 Learn Track</h3>
            <div class="track-preview-card">
              <span class="track-emoji-large">${learnEmoji}</span>
              <div class="track-info">
                <div class="track-topic-large">${learnTopic}</div>
                ${learnHeadline ? `<div class="track-headline">${learnHeadline}</div>` : ''}
                ${learnUniversalTruth ? `<div class="track-truth">${learnUniversalTruth}</div>` : ''}
                <div class="track-assets">
                  ${videoCount > 0 ? `<span class="asset-badge">🎥 ${videoCount} videos</span>` : ''}
                  ${visualCount > 0 ? `<span class="asset-badge">📊 ${visualCount} visuals</span>` : ''}
                  ${phaseCount > 0 ? `<span class="asset-badge">📝 ${phaseCount} phases</span>` : ''}
                </div>
              </div>
            </div>
            
            <!-- Visual Preview -->
            ${visuals.length > 0 ? `
              <div class="visual-preview-section">
                <h4 class="subsection-title">Visual Preview</h4>
                <div class="visual-preview-grid">
                  ${visuals.slice(0, 3).map(v => `
                    <div class="visual-preview-item">
                      <img src="${v.url}" alt="${v.description || 'Lesson visual'}" loading="lazy" 
                           onerror="this.style.display='none'" />
                      ${v.phase ? `<span class="visual-phase-badge">${v.phase}</span>` : ''}
                    </div>
                  `).join('')}
                </div>
              </div>
            ` : ''}
            
            <!-- Copy Preview -->
            ${Object.keys(phases).length > 0 ? `
              <div class="copy-preview-section">
                <h4 class="subsection-title">Lesson Preview</h4>
                <div class="copy-preview-list">
                  ${Object.entries(phases).slice(0, 3).map(([phaseName, phaseData]) => `
                    <div class="copy-preview-item">
                      <div class="copy-phase-name">${phaseName.charAt(0).toUpperCase() + phaseName.slice(1)}</div>
                      <div class="copy-preview-text">${this._truncateText(phaseData.script || '', 150)}</div>
                    </div>
                  `).join('')}
                </div>
              </div>
            ` : ''}
          </div>
          
          <!-- Grow Track Preview -->
          <div class="learner-preview-section">
            <h3 class="section-title">🤖 Grow Track (AI Fluency)</h3>
            ${grow.topic ? `
              <div class="track-preview-card grow">
                <span class="track-emoji-large">${growEmoji}</span>
                <div class="track-info">
                  <div class="track-topic-large">${growTopic}</div>
                  ${grow.objective ? `<div class="track-objective">${grow.objective}</div>` : ''}
                </div>
              </div>
            ` : `
              <div class="track-preview-card grow empty">
                <div class="empty-state">Grow track content coming soon</div>
              </div>
            `}
          </div>
          
          <!-- Phase Preview -->
          <div class="learner-phases-section">
            <h3 class="section-title">What You'll Learn</h3>
            <div class="phases-preview">
              ${this.renderPhasePreviews(phases, visuals, videos)}
            </div>
          </div>
          
          <!-- Start Lesson Button -->
          <div class="learner-actions">
            <a href="/learn.html?day=${day}&track=learn" class="btn-start-lesson">
              Start Lesson →
            </a>
            ${grow?.topic ? `
              <a href="/learn.html?day=${day}&track=grow" class="btn-start-lesson btn-start-lesson-secondary">
                Start Grow Track →
              </a>
            ` : ''}
          </div>
        </div>
      `;
      
      // Update header
      const dayLabel = document.getElementById('audit-day-label');
      const dateLabel = document.getElementById('audit-date-label');
      if (dayLabel) dayLabel.textContent = `Day ${day}`;
      if (dateLabel) dateLabel.textContent = dateStr;
    },

    /**
     * Render phase previews
     */
    renderPhasePreviews(phasesObj, visuals = [], videos = []) {
      const phaseOrder = ['hook', 'question', 'context', 'choice', 'reflection', 'wisdom', 'action'];
      const phaseNames = {
        hook: 'Hook',
        question: 'Question',
        context: 'Context',
        choice: 'Choice',
        reflection: 'Reflection',
        wisdom: 'Wisdom',
        action: 'Action'
      };
      
      return phaseOrder.map((phase, index) => {
        const phaseData = phasesObj[phase] || {};
        const hasVideo = !!phaseData.video || videos.some(v => v.phase === phase);
        const hasVisual = !!phaseData.visual || visuals.some(v => v.phase === phase);
        const hasContent = !!phaseData.script;
        
        return `
          <div class="phase-preview-card ${hasContent ? '' : 'empty'}">
            <div class="phase-number">${index + 1}</div>
            <div class="phase-info">
              <div class="phase-name">${phaseNames[phase]}</div>
              <div class="phase-assets">
                ${hasVideo ? '<span class="asset-indicator video" title="Video available">🎥</span>' : ''}
                ${hasVisual ? '<span class="asset-indicator visual" title="Visual available">📊</span>' : ''}
                ${!hasVideo && !hasVisual && hasContent ? '<span class="asset-indicator text" title="Text content">📝</span>' : ''}
              </div>
            </div>
          </div>
        `;
      }).join('');
    },
    
    /**
     * Truncate text for preview
     */
    _truncateText(text, maxLength) {
      if (!text || text.length <= maxLength) return text || '';
      return text.substring(0, maxLength).trim() + '...';
    },

    /**
     * Render educator view with full copy and visuals
     */
    renderEducatorView() {
      if (!this.auditData) return;
      
      const body = document.getElementById('audit-panel-body');
      if (!body) return;
      
      const day = this.auditData.dayNumber || this.currentDay;
      const audit = this.auditData;
      const learn = audit.tracks?.learn || {};
      const grow = audit.tracks?.grow || {};
      const phases = learn.phases || {};
      const visuals = learn.visuals || [];
      const videos = learn.videos || [];
      
      body.innerHTML = `
        <div class="audit-educator-view">
          <!-- Metadata -->
          <div class="educator-section">
            <h3 class="section-title">Metadata</h3>
            <div class="metadata-grid">
              <div class="metadata-item">
                <span class="metadata-label">Day Number:</span>
                <span class="metadata-value">${day}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Date:</span>
                <span class="metadata-value">${audit.date ? (typeof audit.date === 'string' ? audit.date : audit.date.toLocaleDateString()) : 'N/A'}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Data Source:</span>
                <span class="metadata-value">
                  ${audit.sources?.unified ? '✓ Unified Service' : ''}
                  ${audit.sources?.json ? '✓ JSON' : ''}
                  ${audit.sources?.supabase ? '✓ Supabase' : ''}
                  ${audit.sources?.api ? '✓ API' : ''}
                </span>
              </div>
            </div>
          </div>
          
          <!-- Learn Track Details -->
          <div class="educator-section">
            <h3 class="section-title">📚 Learn Track</h3>
            <div class="track-details">
              <div class="track-detail-item">
                <span class="detail-label">Topic:</span>
                <span class="detail-value">${learn.topic || 'N/A'}</span>
              </div>
              ${learn.headline ? `
                <div class="track-detail-item">
                  <span class="detail-label">Headline:</span>
                  <span class="detail-value">${learn.headline}</span>
                </div>
              ` : ''}
              ${learn.universalTruth ? `
                <div class="track-detail-item">
                  <span class="detail-label">Universal Truth:</span>
                  <span class="detail-value">${learn.universalTruth}</span>
                </div>
              ` : ''}
              <div class="track-detail-item">
                <span class="detail-label">Completeness:</span>
                <span class="detail-value">${learn.completeness || 0}% (${learn.status || 'unknown'})</span>
              </div>
            </div>
          </div>
          
          <!-- Full Copy Display -->
          ${Object.keys(phases).length > 0 ? `
            <div class="educator-section">
              <h3 class="section-title">Full Lesson Copy</h3>
              <div class="full-copy-display">
                ${Object.entries(phases).map(([phaseName, phaseData]) => `
                  <div class="phase-copy-card">
                    <div class="phase-copy-header">
                      <span class="phase-copy-name">${phaseName.charAt(0).toUpperCase() + phaseName.slice(1)}</span>
                      <div class="phase-copy-assets">
                        ${phaseData.video ? '<span class="asset-tag video">🎥 Video</span>' : ''}
                        ${phaseData.visual ? '<span class="asset-tag visual">📊 Visual</span>' : ''}
                        ${phaseData.audio ? '<span class="asset-tag audio">🔊 Audio</span>' : ''}
                      </div>
                    </div>
                    <div class="phase-copy-content">
                      ${phaseData.script ? `<pre class="copy-text">${this._escapeHtml(phaseData.script)}</pre>` : '<span class="no-content">No content</span>'}
                    </div>
                    ${phaseData.video ? `
                      <div class="phase-copy-media">
                        <a href="${phaseData.video}" target="_blank" class="media-link">📹 View Video</a>
                      </div>
                    ` : ''}
                    ${phaseData.visual ? `
                      <div class="phase-copy-media">
                        <img src="${phaseData.visual}" alt="Phase visual" class="phase-visual-preview" loading="lazy" />
                      </div>
                    ` : ''}
                  </div>
                `).join('')}
              </div>
            </div>
          ` : ''}
          
          <!-- Visual Gallery -->
          ${visuals.length > 0 ? `
            <div class="educator-section">
              <h3 class="section-title">Visual Gallery</h3>
              <div class="visual-gallery">
                ${visuals.map((v, idx) => `
                  <div class="gallery-item">
                    <img src="${v.url}" alt="${v.description || `Visual ${idx + 1}`}" loading="lazy" 
                         onerror="this.style.display='none'" />
                    <div class="gallery-item-info">
                      <span class="gallery-phase">${v.phase || 'Unknown'}</span>
                      ${v.description ? `<span class="gallery-desc">${v.description}</span>` : ''}
                    </div>
                  </div>
                `).join('')}
              </div>
            </div>
          ` : ''}
          
          <!-- Video Inventory -->
          ${videos.length > 0 ? `
            <div class="educator-section">
              <h3 class="section-title">Video Inventory</h3>
              <div class="video-list">
                ${videos.map((v, idx) => `
                  <div class="video-item">
                    <span class="video-phase">${v.phase || 'Unknown'}</span>
                    <span class="video-template">${v.template || 'default'}</span>
                    <a href="${v.url}" target="_blank" class="video-link">📹 View</a>
                  </div>
                `).join('')}
              </div>
            </div>
          ` : ''}
          
          <!-- Asset Inventory -->
          <div class="educator-section">
            <h3 class="section-title">Asset Inventory</h3>
            <div class="asset-summary">
              <div class="asset-summary-item">
                <span class="asset-icon">🎥</span>
                <span class="asset-label">Videos:</span>
                <span class="asset-count">${videos.length}</span>
              </div>
              <div class="asset-summary-item">
                <span class="asset-icon">📊</span>
                <span class="asset-label">Visuals:</span>
                <span class="asset-count">${visuals.length}</span>
              </div>
              <div class="asset-summary-item">
                <span class="asset-icon">📝</span>
                <span class="asset-label">Phases:</span>
                <span class="asset-count">${Object.keys(phases).length}</span>
              </div>
              <div class="asset-summary-item">
                <span class="asset-icon">📚</span>
                <span class="asset-label">Atoms:</span>
                <span class="asset-count">${learn.atoms?.length || 0}</span>
              </div>
            </div>
          </div>
          
          <!-- Grow Track -->
          ${grow?.topic ? `
            <div class="educator-section">
              <h3 class="section-title">🤖 Grow Track</h3>
              <div class="track-details">
                <div class="track-detail-item">
                  <span class="detail-label">Topic:</span>
                  <span class="detail-value">${grow.topic}</span>
                </div>
                ${grow.objective ? `
                  <div class="track-detail-item">
                    <span class="detail-label">Objective:</span>
                    <span class="detail-value">${grow.objective}</span>
                  </div>
                ` : ''}
                ${grow.activity ? `
                  <div class="track-detail-item">
                    <span class="detail-label">Activity:</span>
                    <span class="detail-value">${grow.activity}</span>
                  </div>
                ` : ''}
              </div>
            </div>
          ` : ''}
          
          <!-- Errors & Warnings -->
          ${audit.errors?.length > 0 || audit.warnings?.length > 0 ? `
            <div class="educator-section">
              <h3 class="section-title">Issues</h3>
              ${audit.errors?.length > 0 ? `
                <div class="issues-list errors">
                  ${audit.errors.map(e => `<div class="issue-item">⚠️ ${e}</div>`).join('')}
                </div>
              ` : ''}
              ${audit.warnings?.length > 0 ? `
                <div class="issues-list warnings">
                  ${audit.warnings.map(w => `<div class="issue-item">ℹ️ ${w}</div>`).join('')}
                </div>
              ` : ''}
            </div>
          ` : ''}
        </div>
      `;
    },
    
    /**
     * Escape HTML for safe display
     */
    _escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    },

    /**
     * Inject CSS styles
     */
    injectStyles() {
      if (document.getElementById('audit-panel-styles')) return;
      
      const style = document.createElement('style');
      style.id = 'audit-panel-styles';
      style.textContent = `
        .audit-panel {
          position: fixed;
          top: 0;
          right: 0;
          bottom: 0;
          width: 0;
          z-index: 10000;
          transition: width 0.3s ease-out;
          pointer-events: none;
        }
        
        .audit-panel.open {
          width: 500px;
          pointer-events: all;
        }

        /* Prevent body scroll when panel is open */
        body.audit-panel-open {
          overflow: hidden;
        }
        
        .audit-panel-overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          -webkit-backdrop-filter: blur(4px);
          backdrop-filter: blur(4px);
          opacity: 0;
          transition: opacity 0.3s ease-out;
        }
        
        .audit-panel.open .audit-panel-overlay {
          opacity: 1;
        }
        
        .audit-panel-content {
          position: absolute;
          top: 0;
          right: 0;
          bottom: 0;
          width: 500px;
          background: #1a1a1a;
          box-shadow: -4px 0 20px rgba(0, 0, 0, 0.5);
          display: flex;
          flex-direction: column;
          transform: translateX(100%);
          transition: transform 0.3s ease-out;
        }
        
        .audit-panel.open .audit-panel-content {
          transform: translateX(0);
        }
        
        .audit-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px 24px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .audit-panel-title {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        
        #audit-day-label {
          font-size: 1.5em;
          font-weight: 600;
          color: #fff;
        }
        
        .audit-panel-subtitle {
          font-size: 0.9em;
          color: #999;
        }
        
        .audit-panel-close {
          background: none;
          border: none;
          color: #fff;
          font-size: 2em;
          cursor: pointer;
          padding: 0;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
          transition: background 0.2s;
        }
        
        .audit-panel-close:hover {
          background: rgba(255, 255, 255, 0.1);
        }
        
        .audit-view-toggle {
          display: flex;
          gap: 8px;
          padding: 12px 24px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .audit-view-btn {
          flex: 1;
          padding: 8px 16px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          color: #999;
          cursor: pointer;
          transition: all 0.2s;
          font-size: 0.9em;
        }
        
        .audit-view-btn.active {
          background: rgba(37, 99, 235, 0.2);
          border-color: rgba(37, 99, 235, 0.4);
          color: #fff;
        }
        
        .audit-panel-body {
          flex: 1;
          overflow-y: auto;
          padding: 24px;
        }
        
        .audit-loading {
          text-align: center;
          padding: 40px 0;
          color: #999;
        }
        
        /* Learner View Styles */
        .learner-completeness {
          margin-bottom: 24px;
        }
        
        .completeness-gauge-large {
          width: 100%;
          height: 32px;
          background: #333;
          border-radius: 16px;
          overflow: hidden;
          position: relative;
        }
        
        .gauge-fill-large {
          height: 100%;
          transition: width 0.5s ease-out;
          display: flex;
          align-items: center;
          justify-content: flex-end;
          padding-right: 12px;
        }
        
        .gauge-label-large {
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%);
          color: white;
          font-weight: 600;
          font-size: 0.9em;
          text-shadow: 0 0 4px rgba(0,0,0,0.5);
        }
        
        .section-title {
          font-size: 1.1em;
          font-weight: 600;
          margin-bottom: 12px;
          color: #fff;
        }
        
        .learner-preview-section {
          margin-bottom: 24px;
        }
        
        .track-preview-card {
          display: flex;
          gap: 12px;
          padding: 16px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
        }
        
        .track-preview-card.grow {
          border-color: rgba(139, 92, 246, 0.3);
        }
        
        .track-preview-card.empty {
          opacity: 0.5;
        }
        
        .track-emoji-large {
          font-size: 2.5em;
          line-height: 1;
        }
        
        .track-info {
          flex: 1;
        }
        
        .track-topic-large {
          font-size: 1.2em;
          font-weight: 600;
          margin-bottom: 8px;
          color: #fff;
        }
        
        .track-objective {
          font-size: 0.9em;
          color: #999;
          margin-top: 4px;
        }
        
        .track-assets {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-top: 8px;
        }
        
        .asset-badge {
          font-size: 0.75em;
          padding: 4px 8px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          color: #999;
        }
        
        .phases-preview {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 8px;
        }
        
        .phase-preview-card {
          display: flex;
          gap: 8px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          align-items: center;
        }
        
        .phase-preview-card.empty {
          opacity: 0.4;
        }
        
        .phase-number {
          width: 24px;
          height: 24px;
          background: rgba(37, 99, 235, 0.2);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.8em;
          font-weight: 600;
          color: #60a5fa;
        }
        
        .phase-info {
          flex: 1;
        }
        
        .phase-name {
          font-size: 0.9em;
          font-weight: 500;
          color: #fff;
          margin-bottom: 4px;
        }
        
        .phase-assets {
          display: flex;
          gap: 4px;
        }
        
        .asset-indicator {
          font-size: 0.8em;
        }
        
        /* Visual Preview Section */
        .visual-preview-section {
          margin-top: 16px;
        }
        
        .subsection-title {
          font-size: 0.95em;
          font-weight: 500;
          margin-bottom: 8px;
          color: #ccc;
        }
        
        .visual-preview-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        
        .visual-preview-item {
          position: relative;
          aspect-ratio: 1;
          border-radius: 8px;
          overflow: hidden;
          background: rgba(255, 255, 255, 0.05);
        }
        
        .visual-preview-item img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        
        .visual-phase-badge {
          position: absolute;
          bottom: 4px;
          left: 4px;
          font-size: 0.7em;
          padding: 2px 6px;
          background: rgba(0, 0, 0, 0.7);
          border-radius: 4px;
          color: #fff;
        }
        
        /* Copy Preview Section */
        .copy-preview-section {
          margin-top: 16px;
        }
        
        .copy-preview-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .copy-preview-item {
          padding: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 8px;
        }
        
        .copy-phase-name {
          font-size: 0.85em;
          font-weight: 600;
          color: #60a5fa;
          margin-bottom: 4px;
          text-transform: capitalize;
        }
        
        .copy-preview-text {
          font-size: 0.85em;
          color: #999;
          line-height: 1.4;
        }
        
        .track-headline {
          font-size: 0.95em;
          color: #ccc;
          margin-top: 4px;
          font-style: italic;
        }
        
        .track-truth {
          font-size: 0.9em;
          color: #999;
          margin-top: 6px;
          padding-top: 6px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Educator View Styles */
        .track-details {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .track-detail-item {
          display: flex;
          gap: 8px;
          padding: 8px 0;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .detail-label {
          font-weight: 500;
          color: #999;
          min-width: 120px;
        }
        
        .detail-value {
          color: #fff;
          flex: 1;
        }
        
        /* Full Copy Display */
        .full-copy-display {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        
        .phase-copy-card {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 16px;
        }
        
        .phase-copy-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        
        .phase-copy-name {
          font-size: 1em;
          font-weight: 600;
          color: #fff;
          text-transform: capitalize;
        }
        
        .phase-copy-assets {
          display: flex;
          gap: 6px;
        }
        
        .asset-tag {
          font-size: 0.75em;
          padding: 4px 8px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          color: #999;
        }
        
        .phase-copy-content {
          margin-bottom: 12px;
        }
        
        .copy-text {
          font-size: 0.9em;
          color: #ccc;
          line-height: 1.6;
          white-space: pre-wrap;
          word-wrap: break-word;
          background: rgba(0, 0, 0, 0.2);
          padding: 12px;
          border-radius: 8px;
          margin: 0;
          font-family: inherit;
        }
        
        .no-content {
          color: #666;
          font-style: italic;
        }
        
        .phase-copy-media {
          margin-top: 12px;
        }
        
        .media-link {
          display: inline-block;
          padding: 8px 12px;
          background: rgba(37, 99, 235, 0.2);
          border: 1px solid rgba(37, 99, 235, 0.4);
          border-radius: 6px;
          color: #60a5fa;
          text-decoration: none;
          font-size: 0.85em;
          transition: all 0.2s;
        }
        
        .media-link:hover {
          background: rgba(37, 99, 235, 0.3);
        }
        
        .phase-visual-preview {
          max-width: 100%;
          border-radius: 8px;
          margin-top: 8px;
        }
        
        /* Visual Gallery */
        .visual-gallery {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
        }
        
        .gallery-item {
          position: relative;
          aspect-ratio: 16/9;
          border-radius: 8px;
          overflow: hidden;
          background: rgba(255, 255, 255, 0.05);
        }
        
        .gallery-item img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        
        .gallery-item-info {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
          padding: 8px;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        
        .gallery-phase {
          font-size: 0.75em;
          font-weight: 600;
          color: #fff;
          text-transform: capitalize;
        }
        
        .gallery-desc {
          font-size: 0.7em;
          color: #ccc;
        }
        
        /* Video List */
        .video-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .video-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
        }
        
        .video-phase {
          font-size: 0.85em;
          font-weight: 500;
          color: #fff;
          text-transform: capitalize;
          min-width: 80px;
        }
        
        .video-template {
          font-size: 0.8em;
          color: #999;
          flex: 1;
        }
        
        .video-link {
          padding: 6px 12px;
          background: rgba(37, 99, 235, 0.2);
          border: 1px solid rgba(37, 99, 235, 0.4);
          border-radius: 6px;
          color: #60a5fa;
          text-decoration: none;
          font-size: 0.85em;
          transition: all 0.2s;
        }
        
        .video-link:hover {
          background: rgba(37, 99, 235, 0.3);
        }
        
        .learner-actions {
          margin-top: 32px;
          padding-top: 24px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .btn-start-lesson {
          display: block;
          width: 100%;
          padding: 14px 24px;
          background: #2563eb;
          color: white;
          text-align: center;
          text-decoration: none;
          border-radius: 8px;
          font-weight: 600;
          transition: all 0.2s;
        }
        
        .btn-start-lesson:hover {
          background: #1d4ed8;
          transform: translateY(-1px);
        }
        
        .btn-start-lesson-secondary {
          background: rgba(139, 92, 246, 0.2);
          border: 1px solid rgba(139, 92, 246, 0.4);
          color: #a78bfa;
        }
        
        .btn-start-lesson-secondary:hover {
          background: rgba(139, 92, 246, 0.3);
        }
        
        /* Educator View Styles */
        .educator-section {
          margin-bottom: 24px;
          padding-bottom: 24px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .educator-section:last-child {
          border-bottom: none;
        }
        
        .metadata-grid {
          display: grid;
          gap: 12px;
        }
        
        .metadata-item {
          display: flex;
          justify-content: space-between;
          padding: 8px 0;
        }
        
        .metadata-label {
          color: #999;
          font-size: 0.9em;
        }
        
        .metadata-value {
          color: #fff;
          font-weight: 500;
        }
        
        .asset-summary {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
        }
        
        .asset-summary-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
        }
        
        .asset-icon {
          font-size: 1.2em;
        }
        
        .asset-label {
          flex: 1;
          color: #999;
          font-size: 0.9em;
        }
        
        .asset-count {
          color: #fff;
          font-weight: 600;
        }
        
        .tracks-breakdown {
          display: grid;
          gap: 12px;
        }
        
        .track-breakdown-card {
          padding: 16px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
        }
        
        .track-breakdown-card h4 {
          margin: 0 0 8px 0;
          font-size: 1em;
          color: #fff;
        }
        
        .track-stats {
          font-size: 0.9em;
          color: #999;
          line-height: 1.6;
        }
        
        .variants-grid {
          display: grid;
          gap: 8px;
        }
        
        .variant-item {
          display: flex;
          justify-content: space-between;
          padding: 8px 0;
        }
        
        .variant-label {
          color: #999;
        }
        
        .variant-value {
          color: #fff;
          font-weight: 500;
        }
        
        .issues-list {
          margin-top: 12px;
        }
        
        .issues-list.errors .issue-item {
          color: #ef4444;
        }
        
        .issues-list.warnings .issue-item {
          color: #f59e0b;
        }
        
        .issue-item {
          padding: 8px 0;
          font-size: 0.9em;
        }
        
        .empty-state {
          text-align: center;
          padding: 20px;
          color: #666;
          font-size: 0.9em;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
          .audit-panel.open {
            width: 100%;
          }
          
          .audit-panel-content {
            width: 100%;
          }
          
          .phases-preview {
            grid-template-columns: 1fr;
          }
        }
      `;
      
      document.head.appendChild(style);
    }
  };

  // Expose globally
  window.LessonAuditPanel = LessonAuditPanel;
  
  // Also create alias for backward compatibility
  window.LessonAudit = {
    showInspector: (day) => LessonAuditPanel.show(day)
  };
})();

