/**
 * ✨ Kelly Magic Animation System
 * 
 * Creates an enchanting sequence where Kelly:
 * 1. Winks at the viewer (sparkle effect)
 * 2. Snaps her fingers (particle burst)
 * 3. Magic trails to Today's Lesson
 * 4. Lesson card illuminates spectacularly
 * 
 * High-end motion graphics for daily delight.
 */

class KellyMagic {
  constructor() {
    this.heroContainer = null;
    this.heroImage = null;
    this.todayLesson = null;
    this.canvas = null;
    this.ctx = null;
    this.particles = [];
    this.sparkles = [];
    this.animationId = null;
    this.hasPlayed = false;
    
    // Magic sequence timing (ms)
    this.TIMING = {
      initialDelay: 1500,      // Wait for page to settle
      winkStart: 0,            // Relative to sequence start
      winkDuration: 400,
      snapStart: 800,          // After wink
      snapBurstDuration: 600,
      trailStart: 1200,        // Magic trail begins
      trailDuration: 1200,
      revealStart: 2000,       // Lesson illumination
      revealDuration: 800,
      totalSequence: 3500
    };
    
    // Kelly's eye and hand positions (percentages of image)
    // Based on her walking, looking over shoulder pose with director's chair behind
    // She's facing away but looking back at camera with a smile
    this.KELLY_POINTS = {
      eye: { x: 0.55, y: 0.16 },      // Her visible eye (looking back over shoulder)
      hand: { x: 0.62, y: 0.48 }       // Snap position (her right hand near hip)
    };
    
    this.init();
  }
  
  init() {
    // Wait for DOM
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.setup());
    } else {
      this.setup();
    }
  }
  
  setup() {
    this.heroContainer = document.querySelector('.hero-right');
    this.heroImage = document.querySelector('.hero-image');
    this.todayLesson = document.querySelector('.today-lesson');
    
    if (!this.heroContainer || !this.heroImage) {
      console.warn('Kelly Magic: Hero elements not found');
      return;
    }
    
    this.createMagicElements();
    this.createCanvas();
    this.setupIntersectionObserver();
  }
  
  createMagicElements() {
    // Wink sparkle element
    const winkSparkle = document.createElement('div');
    winkSparkle.className = 'kelly-wink-sparkle';
    winkSparkle.innerHTML = `
      <svg viewBox="0 0 100 100" class="wink-star">
        <defs>
          <filter id="wink-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <g filter="url(#wink-glow)">
          <path d="M50 0 L55 40 L100 50 L55 60 L50 100 L45 60 L0 50 L45 40 Z" fill="white"/>
        </g>
      </svg>
      <div class="wink-ring"></div>
    `;
    this.heroContainer.appendChild(winkSparkle);
    this.winkSparkle = winkSparkle;
    
    // Snap burst container
    const snapBurst = document.createElement('div');
    snapBurst.className = 'kelly-snap-burst';
    this.heroContainer.appendChild(snapBurst);
    this.snapBurst = snapBurst;
    
    // Magic connection beam
    const magicBeam = document.createElement('div');
    magicBeam.className = 'kelly-magic-beam';
    document.body.appendChild(magicBeam);
    this.magicBeam = magicBeam;
    
    // Add lesson glow overlay
    if (this.todayLesson) {
      const lessonGlow = document.createElement('div');
      lessonGlow.className = 'lesson-magic-glow';
      this.todayLesson.style.position = 'relative';
      this.todayLesson.insertBefore(lessonGlow, this.todayLesson.firstChild);
      this.lessonGlow = lessonGlow;
    }
  }
  
  createCanvas() {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'kelly-magic-canvas';
    this.canvas.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9999;
    `;
    document.body.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }
  
  resizeCanvas() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }
  
  setupIntersectionObserver() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !this.hasPlayed) {
          setTimeout(() => this.startMagicSequence(), this.TIMING.initialDelay);
          this.hasPlayed = true;
        }
      });
    }, { threshold: 0.3 });
    
    observer.observe(this.heroImage);
  }
  
  getKellyPosition(point) {
    const rect = this.heroImage.getBoundingClientRect();
    return {
      x: rect.left + rect.width * point.x,
      y: rect.top + rect.height * point.y
    };
  }
  
  getLessonPosition() {
    if (!this.todayLesson) return null;
    const rect = this.todayLesson.getBoundingClientRect();
    return {
      x: rect.left + rect.width * 0.5,
      y: rect.top + 50
    };
  }
  
  // ============================================
  // MAGIC SEQUENCE
  // ============================================
  
  startMagicSequence() {
    console.log('✨ Kelly Magic: Starting enchantment sequence');
    
    // Phase 1: The Wink
    setTimeout(() => this.playWink(), this.TIMING.winkStart);
    
    // Phase 2: The Snap
    setTimeout(() => this.playSnap(), this.TIMING.snapStart);
    
    // Phase 3: Magic Trail
    setTimeout(() => this.playMagicTrail(), this.TIMING.trailStart);
    
    // Phase 4: Lesson Reveal
    setTimeout(() => this.playLessonReveal(), this.TIMING.revealStart);
    
    // Cleanup
    setTimeout(() => this.cleanup(), this.TIMING.totalSequence);
  }
  
  // ============================================
  // PHASE 1: THE WINK
  // ============================================
  
  playWink() {
    const eyePos = this.getKellyPosition(this.KELLY_POINTS.eye);
    const heroRect = this.heroContainer.getBoundingClientRect();
    
    // Position wink sparkle relative to hero container
    this.winkSparkle.style.left = `${eyePos.x - heroRect.left}px`;
    this.winkSparkle.style.top = `${eyePos.y - heroRect.top}px`;
    this.winkSparkle.classList.add('active');
    
    // Create secondary sparkles around the wink
    this.createWinkSparkles(eyePos);
    
    setTimeout(() => {
      this.winkSparkle.classList.remove('active');
    }, this.TIMING.winkDuration);
  }
  
  createWinkSparkles(center) {
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const distance = 20 + Math.random() * 30;
      this.sparkles.push({
        x: center.x + Math.cos(angle) * distance,
        y: center.y + Math.sin(angle) * distance,
        size: 2 + Math.random() * 4,
        life: 1,
        decay: 0.02 + Math.random() * 0.02,
        color: this.getSparkleColor(),
        vx: Math.cos(angle) * 0.5,
        vy: Math.sin(angle) * 0.5
      });
    }
    this.startParticleAnimation();
  }
  
  // ============================================
  // PHASE 2: THE SNAP
  // ============================================
  
  playSnap() {
    const handPos = this.getKellyPosition(this.KELLY_POINTS.hand);
    const heroRect = this.heroContainer.getBoundingClientRect();
    
    // Position snap burst
    this.snapBurst.style.left = `${handPos.x - heroRect.left}px`;
    this.snapBurst.style.top = `${handPos.y - heroRect.top}px`;
    this.snapBurst.classList.add('active');
    
    // Create explosion of particles
    this.createSnapParticles(handPos);
    
    // Sound effect indicator (visual pulse)
    this.createSoundWave(handPos);
    
    setTimeout(() => {
      this.snapBurst.classList.remove('active');
    }, this.TIMING.snapBurstDuration);
  }
  
  createSnapParticles(center) {
    const particleCount = 40;
    
    for (let i = 0; i < particleCount; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 3 + Math.random() * 8;
      const size = 3 + Math.random() * 6;
      
      this.particles.push({
        x: center.x,
        y: center.y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        size: size,
        life: 1,
        decay: 0.015 + Math.random() * 0.01,
        color: this.getSnapColor(),
        type: 'snap',
        gravity: 0.1,
        friction: 0.98
      });
    }
    
    // Add special "star" particles
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      this.particles.push({
        x: center.x,
        y: center.y,
        vx: Math.cos(angle) * 12,
        vy: Math.sin(angle) * 12,
        size: 8,
        life: 1,
        decay: 0.025,
        color: '#ffffff',
        type: 'star',
        rotation: Math.random() * Math.PI,
        rotationSpeed: 0.1 + Math.random() * 0.2
      });
    }
  }
  
  createSoundWave(center) {
    for (let i = 0; i < 3; i++) {
      setTimeout(() => {
        this.particles.push({
          x: center.x,
          y: center.y,
          radius: 0,
          maxRadius: 80 + i * 30,
          life: 1,
          decay: 0.03,
          type: 'wave'
        });
      }, i * 100);
    }
  }
  
  // ============================================
  // PHASE 3: MAGIC TRAIL
  // ============================================
  
  playMagicTrail() {
    const handPos = this.getKellyPosition(this.KELLY_POINTS.hand);
    const lessonPos = this.getLessonPosition();
    
    if (!lessonPos) return;
    
    // Animate beam
    this.magicBeam.classList.add('active');
    
    // Create trail particles
    const steps = 30;
    const duration = this.TIMING.trailDuration;
    
    for (let i = 0; i < steps; i++) {
      setTimeout(() => {
        const progress = i / steps;
        // Bezier curve for smooth arc
        const cp1 = { x: handPos.x - 100, y: handPos.y + 200 };
        const cp2 = { x: lessonPos.x + 100, y: lessonPos.y - 100 };
        
        const pos = this.bezierPoint(handPos, cp1, cp2, lessonPos, progress);
        
        // Main trail particle
        this.particles.push({
          x: pos.x,
          y: pos.y,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          size: 6 + Math.random() * 4,
          life: 1,
          decay: 0.02,
          color: this.getMagicColor(),
          type: 'trail',
          glow: true
        });
        
        // Trailing sparkles
        for (let j = 0; j < 3; j++) {
          this.sparkles.push({
            x: pos.x + (Math.random() - 0.5) * 20,
            y: pos.y + (Math.random() - 0.5) * 20,
            size: 2 + Math.random() * 3,
            life: 1,
            decay: 0.04,
            color: this.getSparkleColor()
          });
        }
      }, (i / steps) * duration);
    }
  }
  
  bezierPoint(p0, p1, p2, p3, t) {
    const cx = 3 * (p1.x - p0.x);
    const bx = 3 * (p2.x - p1.x) - cx;
    const ax = p3.x - p0.x - cx - bx;
    
    const cy = 3 * (p1.y - p0.y);
    const by = 3 * (p2.y - p1.y) - cy;
    const ay = p3.y - p0.y - cy - by;
    
    return {
      x: ax * Math.pow(t, 3) + bx * Math.pow(t, 2) + cx * t + p0.x,
      y: ay * Math.pow(t, 3) + by * Math.pow(t, 2) + cy * t + p0.y
    };
  }
  
  // ============================================
  // PHASE 4: LESSON REVEAL
  // ============================================
  
  playLessonReveal() {
    if (!this.todayLesson || !this.lessonGlow) return;
    
    // Activate glow
    this.lessonGlow.classList.add('active');
    this.todayLesson.classList.add('magic-revealed');
    
    // Create the floating "Today's Lesson" attention badge
    this.createAttentionBadge();
    
    // Create celebration particles around lesson
    const rect = this.todayLesson.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Firework burst at center
    this.createFireworkBurst(centerX, rect.top + 60);
    
    // Border sparkles
    for (let i = 0; i < 50; i++) {
      setTimeout(() => {
        const side = Math.floor(Math.random() * 4);
        let x, y;
        
        switch (side) {
          case 0: // top
            x = rect.left + Math.random() * rect.width;
            y = rect.top;
            break;
          case 1: // right
            x = rect.right;
            y = rect.top + Math.random() * rect.height;
            break;
          case 2: // bottom
            x = rect.left + Math.random() * rect.width;
            y = rect.bottom;
            break;
          case 3: // left
            x = rect.left;
            y = rect.top + Math.random() * rect.height;
            break;
        }
        
        this.sparkles.push({
          x: x,
          y: y,
          size: 3 + Math.random() * 4,
          life: 1,
          decay: 0.015,
          color: this.getCelebrationColor(),
          vx: (Math.random() - 0.5) * 4,
          vy: -Math.random() * 3 - 1
        });
      }, i * 20);
    }
    
    // Scroll to lesson section smoothly
    setTimeout(() => {
      this.todayLesson.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
  }
  
  createAttentionBadge() {
    const handPos = this.getKellyPosition(this.KELLY_POINTS.hand);
    const lessonRect = this.todayLesson.getBoundingClientRect();
    
    const badge = document.createElement('div');
    badge.className = 'kelly-attention-badge';
    badge.innerHTML = `
      <span class="badge-sparkle">✨</span>
      <span class="badge-text">Today's Lesson</span>
      <span class="badge-arrow">↓</span>
    `;
    badge.style.cssText = `
      position: fixed;
      left: ${handPos.x}px;
      top: ${handPos.y}px;
      z-index: 10001;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 24px;
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.95), rgba(139, 92, 246, 0.95));
      color: white;
      font-family: 'Inter', sans-serif;
      font-size: 16px;
      font-weight: 600;
      border-radius: 100px;
      box-shadow: 0 8px 32px rgba(59, 130, 246, 0.5), 0 0 60px rgba(139, 92, 246, 0.3);
      transform: translate(-50%, -50%) scale(0);
      opacity: 0;
      pointer-events: none;
    `;
    
    document.body.appendChild(badge);
    
    // Animate badge appearing and moving to lesson
    requestAnimationFrame(() => {
      badge.style.transition = 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
      badge.style.transform = 'translate(-50%, -50%) scale(1)';
      badge.style.opacity = '1';
      
      // Move to lesson area
      setTimeout(() => {
        badge.style.transition = 'all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        badge.style.left = `${lessonRect.left + lessonRect.width / 2}px`;
        badge.style.top = `${lessonRect.top - 40}px`;
      }, 500);
      
      // Fade out
      setTimeout(() => {
        badge.style.transition = 'all 0.5s ease';
        badge.style.opacity = '0';
        badge.style.transform = 'translate(-50%, -100%) scale(0.8)';
      }, 1800);
      
      // Remove
      setTimeout(() => badge.remove(), 2500);
    });
  }
  
  createFireworkBurst(x, y) {
    // Golden celebration burst
    const colors = ['#fbbf24', '#f59e0b', '#fcd34d', '#ffffff', '#3b82f6'];
    
    for (let i = 0; i < 30; i++) {
      const angle = (i / 30) * Math.PI * 2;
      const speed = 4 + Math.random() * 6;
      
      this.particles.push({
        x: x,
        y: y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 2, // Slight upward bias
        size: 4 + Math.random() * 4,
        life: 1,
        decay: 0.02,
        color: colors[Math.floor(Math.random() * colors.length)],
        type: 'firework',
        gravity: 0.15,
        friction: 0.97
      });
    }
    
    // Inner bright core
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      this.sparkles.push({
        x: x + Math.cos(angle) * 20,
        y: y + Math.sin(angle) * 20,
        size: 6,
        life: 1,
        decay: 0.03,
        color: '#ffffff',
        vx: Math.cos(angle) * 8,
        vy: Math.sin(angle) * 8
      });
    }
    
    this.startParticleAnimation();
  }
  
  // ============================================
  // PARTICLE ANIMATION LOOP
  // ============================================
  
  startParticleAnimation() {
    if (this.animationId) return;
    this.animate();
  }
  
  animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Update and draw particles
    this.particles = this.particles.filter(p => {
      p.life -= p.decay;
      if (p.life <= 0) return false;
      
      if (p.type === 'wave') {
        p.radius += 4;
        this.drawWave(p);
      } else if (p.type === 'star') {
        p.x += p.vx;
        p.y += p.vy;
        p.vx *= 0.96;
        p.vy *= 0.96;
        p.rotation += p.rotationSpeed;
        this.drawStar(p);
      } else if (p.type === 'firework') {
        p.x += p.vx;
        p.y += p.vy;
        if (p.gravity) p.vy += p.gravity;
        if (p.friction) {
          p.vx *= p.friction;
          p.vy *= p.friction;
        }
        this.drawFireworkParticle(p);
      } else {
        p.x += p.vx;
        p.y += p.vy;
        if (p.gravity) p.vy += p.gravity;
        if (p.friction) {
          p.vx *= p.friction;
          p.vy *= p.friction;
        }
        this.drawParticle(p);
      }
      
      return true;
    });
    
    // Update and draw sparkles
    this.sparkles = this.sparkles.filter(s => {
      s.life -= s.decay;
      if (s.life <= 0) return false;
      
      if (s.vx) s.x += s.vx;
      if (s.vy) s.y += s.vy;
      
      this.drawSparkle(s);
      return true;
    });
    
    // Continue animation if particles exist
    if (this.particles.length > 0 || this.sparkles.length > 0) {
      this.animationId = requestAnimationFrame(() => this.animate());
    } else {
      this.animationId = null;
    }
  }
  
  drawParticle(p) {
    this.ctx.save();
    this.ctx.globalAlpha = p.life;
    
    if (p.glow) {
      this.ctx.shadowBlur = 15;
      this.ctx.shadowColor = p.color;
    }
    
    this.ctx.fillStyle = p.color;
    this.ctx.beginPath();
    this.ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.restore();
  }
  
  drawSparkle(s) {
    this.ctx.save();
    this.ctx.globalAlpha = s.life;
    this.ctx.fillStyle = s.color;
    this.ctx.shadowBlur = 10;
    this.ctx.shadowColor = s.color;
    
    // Four-point star shape
    const size = s.size * s.life;
    this.ctx.beginPath();
    for (let i = 0; i < 4; i++) {
      const angle = (i / 4) * Math.PI * 2 - Math.PI / 2;
      const innerAngle = angle + Math.PI / 4;
      this.ctx.lineTo(s.x + Math.cos(angle) * size * 2, s.y + Math.sin(angle) * size * 2);
      this.ctx.lineTo(s.x + Math.cos(innerAngle) * size * 0.5, s.y + Math.sin(innerAngle) * size * 0.5);
    }
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.restore();
  }
  
  drawStar(p) {
    this.ctx.save();
    this.ctx.globalAlpha = p.life;
    this.ctx.translate(p.x, p.y);
    this.ctx.rotate(p.rotation);
    this.ctx.fillStyle = p.color;
    this.ctx.shadowBlur = 20;
    this.ctx.shadowColor = p.color;
    
    const size = p.size * p.life;
    this.ctx.beginPath();
    for (let i = 0; i < 4; i++) {
      const angle = (i / 4) * Math.PI * 2;
      const innerAngle = angle + Math.PI / 4;
      this.ctx.lineTo(Math.cos(angle) * size * 2.5, Math.sin(angle) * size * 2.5);
      this.ctx.lineTo(Math.cos(innerAngle) * size * 0.8, Math.sin(innerAngle) * size * 0.8);
    }
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.restore();
  }
  
  drawWave(p) {
    this.ctx.save();
    this.ctx.globalAlpha = p.life * 0.3;
    this.ctx.strokeStyle = '#ffffff';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
    this.ctx.stroke();
    this.ctx.restore();
  }
  
  drawFireworkParticle(p) {
    this.ctx.save();
    this.ctx.globalAlpha = p.life;
    this.ctx.shadowBlur = 20;
    this.ctx.shadowColor = p.color;
    
    // Draw with trail effect
    const trailLength = 3;
    for (let i = 0; i < trailLength; i++) {
      const trailAlpha = (1 - i / trailLength) * p.life * 0.5;
      const trailX = p.x - p.vx * i * 2;
      const trailY = p.y - p.vy * i * 2;
      const trailSize = p.size * p.life * (1 - i / trailLength * 0.5);
      
      this.ctx.globalAlpha = trailAlpha;
      this.ctx.fillStyle = p.color;
      this.ctx.beginPath();
      this.ctx.arc(trailX, trailY, trailSize, 0, Math.PI * 2);
      this.ctx.fill();
    }
    
    // Main particle
    this.ctx.globalAlpha = p.life;
    this.ctx.fillStyle = p.color;
    this.ctx.beginPath();
    this.ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.restore();
  }
  
  // ============================================
  // COLOR PALETTES
  // ============================================
  
  getSparkleColor() {
    const colors = ['#ffffff', '#fef3c7', '#fcd34d', '#fbbf24', '#e0f2fe'];
    return colors[Math.floor(Math.random() * colors.length)];
  }
  
  getSnapColor() {
    const colors = ['#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe', '#ffffff', '#fcd34d'];
    return colors[Math.floor(Math.random() * colors.length)];
  }
  
  getMagicColor() {
    const colors = ['#3b82f6', '#8b5cf6', '#a855f7', '#d946ef', '#f472b6', '#60a5fa'];
    return colors[Math.floor(Math.random() * colors.length)];
  }
  
  getCelebrationColor() {
    const colors = ['#fcd34d', '#fbbf24', '#f59e0b', '#ffffff', '#3b82f6', '#22c55e'];
    return colors[Math.floor(Math.random() * colors.length)];
  }
  
  // ============================================
  // CLEANUP
  // ============================================
  
  cleanup() {
    if (this.magicBeam) {
      this.magicBeam.classList.remove('active');
    }
    
    // Let remaining particles fade naturally
    setTimeout(() => {
      if (this.lessonGlow) {
        this.lessonGlow.classList.add('persistent');
      }
    }, 1000);
  }
  
  // Allow re-triggering for demo purposes
  replay() {
    this.hasPlayed = false;
    this.particles = [];
    this.sparkles = [];
    if (this.lessonGlow) {
      this.lessonGlow.classList.remove('active', 'persistent');
    }
    if (this.todayLesson) {
      this.todayLesson.classList.remove('magic-revealed');
    }
    setTimeout(() => this.startMagicSequence(), 100);
  }
}

// Initialize on load
const kellyMagic = new KellyMagic();

// Expose for debugging/demo
window.kellyMagic = kellyMagic;

// Add a subtle replay hint after first play
setTimeout(() => {
  if (kellyMagic.hasPlayed) {
    console.log('✨ Tip: Run kellyMagic.replay() in console to see the magic again!');
  }
}, 5000);

// Create floating replay button (only in development or with URL param)
if (window.location.search.includes('debug') || window.location.hostname === 'localhost') {
  document.addEventListener('DOMContentLoaded', () => {
    const replayBtn = document.createElement('button');
    replayBtn.innerHTML = '✨ Replay Magic';
    replayBtn.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 20px;
      z-index: 10000;
      padding: 12px 20px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      color: white;
      border: none;
      border-radius: 100px;
      font-family: inherit;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
      transition: all 0.3s ease;
    `;
    replayBtn.onmouseover = () => replayBtn.style.transform = 'scale(1.05)';
    replayBtn.onmouseout = () => replayBtn.style.transform = 'scale(1)';
    replayBtn.onclick = () => kellyMagic.replay();
    document.body.appendChild(replayBtn);
  });
}

