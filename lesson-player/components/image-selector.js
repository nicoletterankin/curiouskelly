/**
 * Image Selector Component
 * Emotion-based image selection for Kelly expressions
 * 
 * Maps lesson phases and interaction types to Kelly expression images
 */

class ImageSelector {
    constructor() {
        // Image mapping: expression name -> image file path
        this.imageMap = {
            'curious': '../lessons/images/kelly-directors-chair-curious.png',
            'explaining': '../lessons/images/kelly-directors-chair-explaining.png',
            'celebrating': '../lessons/images/kelly-directors-chair-celebrating.png',
            'listening': '../lessons/images/kelly-directors-chair-listening.png',
            'wisdom': '../lessons/images/kelly-directors-chair-wisdom.png'
        };
        
        // Default fallback image
        this.defaultImage = this.imageMap['curious'];
        
        // Phase to expression mapping
        this.phaseMapping = {
            'welcome': 'curious',
            'teaching': 'explaining',
            'mainContent': 'explaining',
            'practice': 'listening',
            'wisdom': 'wisdom',
            'wisdomMoment': 'wisdom',
            'reflection': 'wisdom'
        };
        
        // Interaction type to expression mapping
        this.interactionMapping = {
            'question': 'curious',
            'explanation': 'explaining',
            'celebration': 'celebrating',
            'response': 'listening',
            'feedback': 'listening',
            'wisdom': 'wisdom'
        };
        
        // Response sentiment to expression mapping
        this.sentimentMapping = {
            'positive': 'celebrating',
            'correct': 'celebrating',
            'encouraging': 'celebrating',
            'neutral': 'listening',
            'thoughtful': 'listening',
            'reflective': 'wisdom'
        };
    }
    
    /**
     * Get image path based on current lesson state
     * @param {Object} state - Current lesson state
     * @param {string} state.phase - Current phase (welcome, teaching, wisdom, etc.)
     * @param {string} state.interactionType - Type of interaction (question, explanation, etc.)
     * @param {string} state.sentiment - Response sentiment (positive, neutral, etc.)
     * @param {Object} state.learnerResponse - Learner's response data
     * @returns {string} Path to image file
     */
    selectImage(state) {
        if (!state) {
            return this.defaultImage;
        }
        
        const { phase, interactionType, sentiment, learnerResponse } = state;
        
        // Priority 1: Explicit expression override (if provided)
        if (state.expression && this.imageMap[state.expression]) {
            return this.imageMap[state.expression];
        }
        
        // Priority 2: Sentiment-based selection (for responses)
        if (sentiment && this.sentimentMapping[sentiment]) {
            const expression = this.sentimentMapping[sentiment];
            if (this.imageMap[expression]) {
                return this.imageMap[expression];
            }
        }
        
        // Priority 3: Interaction type-based selection
        if (interactionType && this.interactionMapping[interactionType]) {
            const expression = this.interactionMapping[interactionType];
            if (this.imageMap[expression]) {
                return this.imageMap[expression];
            }
        }
        
        // Priority 4: Phase-based selection (default)
        if (phase) {
            // Normalize phase name
            const normalizedPhase = phase.toLowerCase().replace(/\s+/g, '');
            const expression = this.phaseMapping[normalizedPhase] || this.phaseMapping[phase];
            
            if (expression && this.imageMap[expression]) {
                return this.imageMap[expression];
            }
        }
        
        // Fallback to default
        return this.defaultImage;
    }
    
    /**
     * Get expression name for a given state (useful for debugging)
     * @param {Object} state - Current lesson state
     * @returns {string} Expression name
     */
    getExpressionName(state) {
        if (!state) {
            return 'curious';
        }
        
        const { phase, interactionType, sentiment } = state;
        
        if (state.expression) {
            return state.expression;
        }
        
        if (sentiment && this.sentimentMapping[sentiment]) {
            return this.sentimentMapping[sentiment];
        }
        
        if (interactionType && this.interactionMapping[interactionType]) {
            return this.interactionMapping[interactionType];
        }
        
        if (phase) {
            const normalizedPhase = phase.toLowerCase().replace(/\s+/g, '');
            return this.phaseMapping[normalizedPhase] || this.phaseMapping[phase] || 'curious';
        }
        
        return 'curious';
    }
    
    /**
     * Update image mapping (allows dynamic updates)
     * @param {Object} newMap - New image mapping object
     */
    updateImageMap(newMap) {
        this.imageMap = { ...this.imageMap, ...newMap };
    }
    
    /**
     * Get all available expressions
     * @returns {Array<string>} List of expression names
     */
    getAvailableExpressions() {
        return Object.keys(this.imageMap);
    }
    
    /**
     * Check if an image exists for an expression
     * @param {string} expression - Expression name
     * @returns {boolean} True if image exists
     */
    hasImage(expression) {
        return !!this.imageMap[expression];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ImageSelector;
}




