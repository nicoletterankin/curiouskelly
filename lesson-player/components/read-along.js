/**
 * Read-Along Component
 * 
 * Displays synchronized text with audio playback.
 * Highlights current word/phrase as audio plays.
 * 
 * Features:
 * - Word-level sync markers
 * - Smooth transitions
 * - Age-appropriate styling
 * - Touch/click to jump to timestamp
 */

class ReadAlongComponent {
  constructor(containerElement) {
    this.container = containerElement;
    this.text = '';
    this.syncMarkers = [];
    this.currentMarkerIndex = -1;
    this.audioElement = null;
    this.enabled = true;
  }

  /**
   * Initialize read-along with text and sync markers
   * @param {string} text - Full text content
   * @param {Array} syncMarkers - Array of {word, startTime, endTime}
   * @param {HTMLAudioElement} audioElement - Associated audio element
   */
  initialize(text, syncMarkers, audioElement) {
    this.text = text;
    this.syncMarkers = syncMarkers;
    this.audioElement = audioElement;
    this.currentMarkerIndex = -1;

    this.render();
    this.setupEventListeners();
  }

  /**
   * Render the read-along component
   */
  render() {
    if (!this.enabled || !this.text) {
      this.container.innerHTML = '';
      return;
    }

    // If we have sync markers, render word-by-word
    if (this.syncMarkers && this.syncMarkers.length > 0) {
      this.container.innerHTML = this.syncMarkers.map((marker, index) => 
        `<span class="read-along-word" data-index="${index}" data-start="${marker.startTime}" data-end="${marker.endTime}">
          ${this.escapeHtml(marker.word)}
        </span>`
      ).join(' ');
    } else {
      // Fallback: show text without sync
      this.container.innerHTML = `<p class="read-along-text">${this.escapeHtml(this.text)}</p>`;
    }
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    if (!this.audioElement) return;

    // Update highlighting as audio plays
    this.audioElement.addEventListener('timeupdate', () => {
      this.updateHighlight(this.audioElement.currentTime);
    });

    // Allow clicking words to jump to timestamp
    this.container.addEventListener('click', (e) => {
      const wordElement = e.target.closest('.read-along-word');
      if (wordElement && this.audioElement) {
        const startTime = parseFloat(wordElement.dataset.start);
        this.audioElement.currentTime = startTime;
        if (this.audioElement.paused) {
          this.audioElement.play();
        }
      }
    });

    // Reset highlighting when audio ends
    this.audioElement.addEventListener('ended', () => {
      this.clearHighlight();
    });

    // Reset highlighting when audio is paused
    this.audioElement.addEventListener('pause', () => {
      this.clearHighlight();
    });
  }

  /**
   * Update word highlighting based on current time
   * @param {number} currentTime - Current audio playback time in seconds
   */
  updateHighlight(currentTime) {
    if (!this.syncMarkers || this.syncMarkers.length === 0) return;

    // Find the marker that matches current time
    const markerIndex = this.syncMarkers.findIndex((marker, index) => {
      return currentTime >= marker.startTime && currentTime < marker.endTime;
    });

    if (markerIndex !== this.currentMarkerIndex) {
      this.setHighlight(markerIndex);
    }
  }

  /**
   * Set highlight on specific word
   * @param {number} index - Marker index to highlight
   */
  setHighlight(index) {
    // Clear previous highlight
    this.clearHighlight();

    if (index >= 0 && index < this.syncMarkers.length) {
      const wordElement = this.container.querySelector(`.read-along-word[data-index="${index}"]`);
      if (wordElement) {
        wordElement.classList.add('read-along-active');
        this.currentMarkerIndex = index;

        // Auto-scroll to keep current word visible
        this.scrollToWord(wordElement);
      }
    }
  }

  /**
   * Clear all highlights
   */
  clearHighlight() {
    const activeWords = this.container.querySelectorAll('.read-along-active');
    activeWords.forEach(word => word.classList.remove('read-along-active'));
    this.currentMarkerIndex = -1;
  }

  /**
   * Scroll container to keep word visible
   * @param {HTMLElement} wordElement - Word element to scroll to
   */
  scrollToWord(wordElement) {
    const container = this.container;
    const wordRect = wordElement.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    // Check if word is outside visible area
    if (wordRect.bottom > containerRect.bottom || wordRect.top < containerRect.top) {
      wordElement.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  }

  /**
   * Toggle read-along enabled state
   * @param {boolean} enabled - Whether to enable read-along
   */
  setEnabled(enabled) {
    this.enabled = enabled;
    this.render();
  }

  /**
   * Escape HTML in text
   * @param {string} text - Text to escape
   * @returns {string} Escaped text
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Clear component
   */
  clear() {
    this.text = '';
    this.syncMarkers = [];
    this.currentMarkerIndex = -1;
    this.container.innerHTML = '';
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ReadAlongComponent;
}



