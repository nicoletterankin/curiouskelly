/**
 * ✨ TRUST & SAFETY - SIMULATED SOCIAL CONTENT CONTROLS
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * This module manages the user's preferences for simulated social content.
 * It enforces the "Trust & Safety" rules:
 * 1. Master toggle to disable all simulated content.
 * 2. Clear visual marking (✨) for all simulated content.
 * 3. Transparent disclosure via tooltips/modals.
 * 
 * @module SimulatedContent
 */

(function(global) {
    const STORAGE_KEY = 'simulatedContentPrefs';
    const EVENT_NAME = 'simulated-content-changed';

    // Default preferences
    const DEFAULT_PREFS = {
        enabled: true,           // Master toggle
        showIndicators: true,    // Show ✨ icons
        showTooltips: true,      // Show explanatory tooltips
        types: {
            peerComments: true,
            ageResponses: true,
            questions: true,
            milestones: true,
            discussions: true
        }
    };

    class SimulatedContentManager {
        constructor() {
            this.prefs = this._loadPrefs();
            this._applyStateToDOM();
            console.log('✨ Simulated Content Manager initialized');
        }

        /**
         * Load preferences from localStorage or use defaults
         */
        _loadPrefs() {
            try {
                const stored = localStorage.getItem(STORAGE_KEY);
                return stored ? { ...DEFAULT_PREFS, ...JSON.parse(stored) } : { ...DEFAULT_PREFS };
            } catch (e) {
                console.warn('Failed to load simulated content prefs', e);
                return { ...DEFAULT_PREFS };
            }
        }

        /**
         * Save current preferences to localStorage
         */
        _savePrefs() {
            try {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(this.prefs));
                this._applyStateToDOM();
                this._dispatchChange();
            } catch (e) {
                console.error('Failed to save simulated content prefs', e);
            }
        }

        /**
         * Apply global classes to the document body based on preferences.
         * CSS can use these classes to hide/show elements.
         */
        _applyStateToDOM() {
            const body = document.body;
            if (this.prefs.enabled) {
                body.classList.remove('simulated-content-disabled');
                body.classList.add('simulated-content-enabled');
            } else {
                body.classList.remove('simulated-content-enabled');
                body.classList.add('simulated-content-disabled');
            }

            if (this.prefs.showIndicators) {
                body.classList.add('simulated-indicators-visible');
            } else {
                body.classList.remove('simulated-indicators-visible');
            }
        }

        /**
         * Dispatch a custom event when preferences change
         */
        _dispatchChange() {
            const event = new CustomEvent(EVENT_NAME, { detail: this.prefs });
            window.dispatchEvent(event);
        }

        // ═════════════════════════════════════════════════════════════════════════
        // PUBLIC API
        // ═════════════════════════════════════════════════════════════════════════

        /**
         * Master toggle for all simulated content
         * @param {boolean} enabled 
         */
        toggle(enabled) {
            this.prefs.enabled = !!enabled;
            this._savePrefs();
            console.log(`✨ Simulated content ${this.prefs.enabled ? 'ENABLED' : 'DISABLED'}`);
            return this.prefs.enabled;
        }

        /**
         * Check if a specific type of simulated content is allowed
         * @param {string} type - e.g., 'peerComments', 'questions'
         */
        isAllowed(type) {
            if (!this.prefs.enabled) return false;
            if (type && this.prefs.types[type] === false) return false;
            return true;
        }

        /**
         * Get current preferences
         */
        getPrefs() {
            return { ...this.prefs };
        }

        /**
         * Render the standardized disclosure HTML
         * @returns {string} HTML string for the indicator
         */
        getIndicatorHTML() {
            return `
                <span class="simulated-indicator" title="Simulated learner perspective">
                    ✨
                    <span class="simulated-tooltip">
                        <strong>Simulated Learner</strong><br>
                        This comment was created to show diverse learning perspectives.<br>
                        <a href="#" onclick="window.KellySimulatedContent.toggle(false); return false;">Turn off simulated content</a>
                    </span>
                </span>
            `;
        }
    }

    // Initialize and expose globally
    global.KellySimulatedContent = new SimulatedContentManager();

    // Alias for backward compatibility/ease of use if 'kelly' object exists
    if (global.kelly) {
        global.kelly.toggleSimulatedContent = (enabled) => global.KellySimulatedContent.toggle(enabled);
    } else {
        // If kelly object doesn't exist yet, we can't attach it directly, 
        // but the main app initialization should check for KellySimulatedContent.
        // We'll also set up a temporary kelly object just in case.
        global.kelly = global.kelly || {};
        global.kelly.toggleSimulatedContent = (enabled) => global.KellySimulatedContent.toggle(enabled);
    }

})(window);








