/**
 * Kelly 2D Avatar System
 * 
 * Clean, professional 2D avatar system using real Kelly images
 * 5-phase Hot-or-Not learning journey
 */

export class Kelly2DAvatar {
  constructor(containerElement) {
    this.container = containerElement;
    this.currentPhase = 'welcome';
    this.currentExpression = 'welcome';
    this.isTransitioning = false;
    
    // Map phases to available Kelly images
    this.expressions = {
      // Welcome phase
      welcome: 'kelly-chair-curious.png',
      
      // Question phases
      question: 'kelly-chair-side.png',
      curious: 'kelly-chair-side.png',
      
      // Hot reactions (explaining/teaching)
      explaining: 'kelly-headshot-neutral.png',
      teaching: 'kelly-headshot-neutral.png',
      
      // Not reactions (celebrating)
      celebrating: 'kelly-headshot-smile.png',
      excited: 'kelly-headshot-smile.png',
      
      // Wisdom phase
      wisdom: 'kelly-headshot-smile.png',
      serene: 'kelly-headshot-neutral.png'
    };
    
    this.init();
  }
  
  init() {
    this.createAvatarDOM();
    this.preloadImages();
    console.log('[Kelly2D] Initialized with real Kelly assets');
  }
  
  createAvatarDOM() {
    this.container.innerHTML = `
      <div class="kelly-2d-wrapper" data-phase="${this.currentPhase}">
        <!-- Current image -->
        <img 
          class="kelly-image active" 
          id="kelly-current"
          src="/kelly-images/kelly-welcome-chair.png"
          alt="Kelly"
          data-expression="welcome"
        />
        
        <!-- Next image (for crossfade) -->
        <img 
          class="kelly-image inactive" 
          id="kelly-next"
          src=""
          alt=""
        />
        
        <!-- Minimal state indicator -->
        <div class="kelly-state-badge" id="kelly-state">
          <span>Welcome</span>
        </div>
      </div>
    `;
    
    this.elements = {
      wrapper: this.container.querySelector('.kelly-2d-wrapper'),
      current: this.container.querySelector('#kelly-current'),
      next: this.container.querySelector('#kelly-next'),
      stateBadge: this.container.querySelector('#kelly-state')
    };
  }
  
  preloadImages() {
    // Preload the key images we'll need
    const imagesToPreload = [
      '/kelly-images/kelly-welcome-chair.png',
      '/kelly-images/kelly-question-left.png',
      '/kelly-images/kelly-celebrating-smile.png',
      '/kelly-images/kelly-explaining-neutral.png'
    ];
    
    imagesToPreload.forEach(src => {
      const img = new Image();
      img.src = src;
    });
  }
  
  /**
   * Set the lesson phase
   */
  async setPhase(phase, choice = null) {
    if (this.isTransitioning) return;
    
    console.log(`[Kelly2D] Phase: ${this.currentPhase} → ${phase}`, { choice });
    
    this.currentPhase = phase;
    this.elements.wrapper.setAttribute('data-phase', phase);
    
    // Determine which expression to show
    let expression = this.getExpressionForPhase(phase, choice);
    
    // Update state badge
    this.updateStateBadge(phase, choice);
    
    // Transition to new expression
    await this.transitionTo(expression);
    
    // Emit event
    document.dispatchEvent(new CustomEvent('kelly-phase-changed', {
      detail: { phase, expression }
    }));
  }
  
  getExpressionForPhase(phase, choice) {
    // Map phases to expressions
    const phaseMap = {
      'welcome': 'welcome',
      'q1': 'question',
      'q2': 'question',
      'q3': 'question',
      'wisdom': 'wisdom'
    };
    
    // Handle reactions
    if (choice) {
      if (choice === 'a') {
        return 'explaining'; // Hot reaction
      } else if (choice === 'b') {
        return 'celebrating'; // Not reaction
      }
    }
    
    return phaseMap[phase] || 'welcome';
  }
  
  updateStateBadge(phase, choice) {
    const labels = {
      'welcome': 'Welcome',
      'q1': 'Question 1',
      'q2': 'Question 2',
      'q3': 'Question 3',
      'wisdom': 'Wisdom'
    };
    
    let label = labels[phase] || phase;
    
    if (choice) {
      label += choice === 'a' ? ' - Explaining' : ' - Celebrating!';
    }
    
    this.elements.stateBadge.querySelector('span').textContent = label;
  }
  
  /**
   * Smooth crossfade transition between images
   */
  async transitionTo(expression) {
    if (this.currentExpression === expression) return;
    if (this.isTransitioning) return;
    
    this.isTransitioning = true;
    
    // Get image path for expression
    const imagePath = this.getImagePath(expression);
    
    // Load next image
    this.elements.next.src = imagePath;
    this.elements.next.setAttribute('data-expression', expression);
    
    // Wait for image to load
    await this.waitForImageLoad(this.elements.next);
    
    // Crossfade
    this.elements.current.classList.remove('active');
    this.elements.current.classList.add('inactive');
    this.elements.next.classList.remove('inactive');
    this.elements.next.classList.add('active');
    
    // Wait for transition
    await this.wait(600);
    
    // Swap references
    const temp = this.elements.current;
    this.elements.current = this.elements.next;
    this.elements.next = temp;
    
    this.currentExpression = expression;
    this.isTransitioning = false;
  }
  
  getImagePath(expression) {
    // Map expressions to actual Kelly image files
    const imageMap = {
      'welcome': '/kelly-images/kelly-welcome-chair.png',
      'question': '/kelly-images/kelly-question-left.png',
      'curious': '/kelly-images/kelly-question-left.png',
      'explaining': '/kelly-images/kelly-explaining-neutral.png',
      'teaching': '/kelly-images/kelly-explaining-neutral.png',
      'celebrating': '/kelly-images/kelly-celebrating-smile.png',
      'excited': '/kelly-images/kelly-celebrating-smile.png',
      'wisdom': '/kelly-images/kelly-celebrating-smile.png',
      'serene': '/kelly-images/kelly-explaining-neutral.png'
    };
    
    return imageMap[expression] || imageMap['welcome'];
  }
  
  waitForImageLoad(img) {
    return new Promise((resolve) => {
      if (img.complete) {
        resolve();
      } else {
        img.onload = () => resolve();
        img.onerror = () => resolve(); // Continue even if error
      }
    });
  }
  
  wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  /**
   * Public API
   */
  
  showWelcome() {
    this.setPhase('welcome');
  }
  
  showQuestion(number) {
    this.setPhase(`q${number}`);
  }
  
  showReaction(questionNumber, choice) {
    this.setPhase(`q${questionNumber}`, choice);
    
    // Auto-advance after reaction
    setTimeout(() => {
      if (questionNumber < 3) {
        this.showQuestion(questionNumber + 1);
      } else {
        this.showWisdom();
      }
    }, 3000);
  }
  
  showWisdom() {
    this.setPhase('wisdom');
  }
  
  destroy() {
    console.log('[Kelly2D] Destroyed');
  }
}

export default Kelly2DAvatar;

