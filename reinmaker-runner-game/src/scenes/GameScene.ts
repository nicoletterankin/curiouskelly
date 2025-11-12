import Phaser from 'phaser';

export default class GameScene extends Phaser.Scene {
  private player!: Phaser.Physics.Arcade.Sprite;
  private ground!: Phaser.GameObjects.TileSprite;
  private background!: Phaser.GameObjects.TileSprite;
  private groundStripe!: Phaser.GameObjects.TileSprite;
  
  private obstacles!: Phaser.Physics.Arcade.Group;
  private stones!: Phaser.Physics.Arcade.Group;
  
  private score: number = 0;
  private stonesCollected: number = 0;
  private scoreText!: Phaser.GameObjects.Text;
  private stonesText!: Phaser.GameObjects.Text;
  private comboText!: Phaser.GameObjects.Text;
  private speedText!: Phaser.GameObjects.Text;
  
  private gameSpeed: number = 200;
  private isGameOver: boolean = false;
  private isPaused: boolean = false;
  
  private spawnTimer: number = 0;
  private stoneTimer: number = 0;
  
  // Jump physics
  private coyoteTime: number = 0;
  private coyoteTimeMax: number = 100; // ms grace period after leaving ground
  private jumpBufferTime: number = 0;
  private jumpBufferMax: number = 150; // ms buffer before landing
  private isJumping: boolean = false;
  private wasOnGround: boolean = false;
  
  // Combo system
  private comboCount: number = 0;
  private comboMultiplier: number = 1;
  private lastStoneTime: number = 0;
  private comboTimeout: number = 2000; // ms to maintain combo
  private bestCombo: number = 0;
  
  // Visual effects
  private stoneGlowTimers: Map<Phaser.GameObjects.Sprite, Phaser.Time.TimerEvent> = new Map();
  
  // Audio (using Web Audio API for simple tones)
  private audioContext!: AudioContext;
  
  // High score
  private highScore: number = 0;
  private highScoreText!: Phaser.GameObjects.Text;
  
  // Pattern system
  private lastPattern: string = '';
  private patternCounter: number = 0;

  constructor() {
    super({ key: 'GameScene' });
  }

  create() {
    const { width, height } = this.cameras.main;
    
    // Initialize audio context
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    } catch (e) {
      console.warn('Web Audio API not supported');
    }
    
    // Load high score
    this.highScore = parseInt(localStorage.getItem('reinmakerHighScore') || '0', 10);
    
    // Background (scrolling slowly)
    this.background = this.add.tileSprite(0, 0, width * 2, 256, 'bg')
      .setOrigin(0, 0)
      .setScale(1.5);
    
    // Ground
    const groundY = height - 80;
    this.ground = this.add.tileSprite(0, groundY, width * 2, 64, 'ground_tex')
      .setOrigin(0, 0);
    
    // Ground stripe (animated)
    this.groundStripe = this.add.tileSprite(0, groundY - 3, width * 2, 6, 'ground_stripe')
      .setOrigin(0, 0)
      .setAlpha(0.8);
    
    // Create particle sprite (simple white circle) - must be done first
    const graphics = this.add.graphics();
    graphics.fillStyle(0xffffff);
    graphics.fillCircle(0, 0, 4);
    graphics.generateTexture('particle', 8, 8);
    graphics.destroy();
    
    // Player
    this.player = this.physics.add.sprite(150, groundY - 50, 'player')
      .setScale(0.15)
      .setCollideWorldBounds(true);
    
    // Set player hitbox (smaller than sprite for better gameplay)
    this.player.setSize(this.player.width * 0.6, this.player.height * 0.8);
    
    // Player bounce animation (since we don't have run frames yet)
    this.tweens.add({
      targets: this.player,
      y: `-=${5}`,
      duration: 300,
      yoyo: true,
      repeat: -1,
      ease: 'Sine.easeInOut'
    });
    
    // Obstacles group
    this.obstacles = this.physics.add.group();
    
    // Stones group
    this.stones = this.physics.add.group();
    
    // Collisions
    this.physics.add.collider(this.player, this.obstacles, this.hitObstacle, undefined, this);
    this.physics.add.overlap(this.player, this.stones, this.collectStone, undefined, this);
    
    // HUD
    this.scoreText = this.add.text(16, 16, 'Score: 0', {
      fontSize: '24px',
      color: '#F2F7FA',
      fontStyle: 'bold',
      stroke: '#000000',
      strokeThickness: 2
    }).setScrollFactor(0);
    
    this.stonesText = this.add.text(16, 48, 'Stones: 0', {
      fontSize: '20px',
      color: '#0BB39C',
      stroke: '#000000',
      strokeThickness: 2
    }).setScrollFactor(0);
    
    this.comboText = this.add.text(16, 76, '', {
      fontSize: '18px',
      color: '#FFE066',
      fontStyle: 'bold',
      stroke: '#000000',
      strokeThickness: 2
    }).setScrollFactor(0);
    
    this.speedText = this.add.text(width - 120, 16, 'Speed: 1.0x', {
      fontSize: '16px',
      color: '#D8A24A',
      stroke: '#000000',
      strokeThickness: 2
    }).setScrollFactor(0);
    
    this.highScoreText = this.add.text(16, 100, `Best: ${this.highScore}`, {
      fontSize: '16px',
      color: '#D8A24A',
      stroke: '#000000',
      strokeThickness: 2
    }).setScrollFactor(0);
    
    // Controls
    this.input.keyboard?.on('keydown-SPACE', () => this.jump());
    this.input.keyboard?.on('keydown-UP', () => this.jump());
    this.input.keyboard?.on('keyup-SPACE', () => this.onJumpRelease());
    this.input.keyboard?.on('keyup-UP', () => this.onJumpRelease());
    this.input.keyboard?.on('keydown-ESC', () => this.togglePause());
    this.input.on('pointerdown', () => {
      if (!this.isPaused) {
        this.jump();
      }
    });
    this.input.on('pointerup', () => this.onJumpRelease());
    
    // Restart on R key
    this.input.keyboard?.on('keydown-R', () => {
      if (this.isGameOver) {
        this.scene.restart();
      }
    });
  }

  update(_time: number, delta: number) {
    if (this.isGameOver || this.isPaused) return;
    
    // Update jump physics timers
    const isOnGround = this.player.body?.touching.down || false;
    
    if (isOnGround) {
      this.coyoteTime = this.coyoteTimeMax;
      this.isJumping = false;
      this.wasOnGround = true;
      
      // Process jump buffer
      if (this.jumpBufferTime > 0) {
        this.jumpBufferTime = 0;
        this.executeJump();
      }
    } else {
      if (this.wasOnGround) {
        this.coyoteTime = this.coyoteTimeMax;
        this.wasOnGround = false;
      }
      this.coyoteTime = Math.max(0, this.coyoteTime - delta);
    }
    
    // Decay jump buffer
    if (this.jumpBufferTime > 0) {
      this.jumpBufferTime = Math.max(0, this.jumpBufferTime - delta);
    }
    
    // Update combo timer
    if (this.comboCount > 0 && _time - this.lastStoneTime > this.comboTimeout) {
      this.comboCount = 0;
      this.comboMultiplier = 1;
      this.comboText.setText('');
    }
    
    // Scroll background and ground
    this.background.tilePositionX += this.gameSpeed * delta / 1000 * 0.3;
    this.ground.tilePositionX += this.gameSpeed * delta / 1000;
    this.groundStripe.tilePositionX += this.gameSpeed * delta / 1000 * 2; // Faster stripe
    
    // Update score
    this.score += delta / 100;
    const scoreInt = Math.floor(this.score);
    this.scoreText.setText(`Score: ${scoreInt}`);
    
    // Update speed display
    const speedMultiplier = (this.gameSpeed / 200).toFixed(1);
    this.speedText.setText(`Speed: ${speedMultiplier}x`);
    
    // Update high score if beaten
    if (scoreInt > this.highScore) {
      this.highScore = scoreInt;
      this.highScoreText.setText(`Best: ${this.highScore}`);
      this.highScoreText.setColor('#FFE066'); // Gold color for new record
    }
    
    // Gradually increase difficulty
    this.gameSpeed += delta / 10000;
    
    // Spawn obstacles (dynamic timing with patterns)
    this.spawnTimer += delta;
    const obstacleDelay = Math.max(600, 2000 - (this.score / 1000) * 100);
    if (this.spawnTimer > obstacleDelay) {
      this.spawnObstaclePattern();
      this.spawnTimer = 0;
    }
    
    // Spawn stones (dynamic timing)
    this.stoneTimer += delta;
    const stoneDelay = Math.max(400, 1500 - (this.score / 1500) * 100);
    if (this.stoneTimer > stoneDelay) {
      this.spawnStone();
      this.stoneTimer = 0;
    }
    
    // Clean up off-screen objects
    this.obstacles.children.entries.forEach((obstacle) => {
      const sprite = obstacle as Phaser.Physics.Arcade.Sprite;
      if (sprite.x < -100) {
        sprite.destroy();
      }
    });
    
    this.stones.children.entries.forEach((stone) => {
      const sprite = stone as Phaser.Physics.Arcade.Sprite;
      if (sprite.x < -100) {
        sprite.destroy();
        // Clean up glow timer
        const timer = this.stoneGlowTimers.get(sprite);
        if (timer) {
          timer.destroy();
          this.stoneGlowTimers.delete(sprite);
        }
      }
    });
  }

  private jump() {
    if (this.isGameOver || this.isPaused) return;
    
    const isOnGround = this.player.body?.touching.down || false;
    
    if (isOnGround || this.coyoteTime > 0) {
      this.executeJump();
    } else {
      // Jump buffer - queue jump for when we land
      this.jumpBufferTime = this.jumpBufferMax;
    }
  }
  
  private executeJump() {
    this.isJumping = true;
    this.coyoteTime = 0;
    
    // Variable jump height based on hold time (will be enhanced on release)
    this.player.setVelocityY(-500);
    
    // Visual feedback: scale up slightly
    this.tweens.add({
      targets: this.player,
      scaleX: 0.17,
      scaleY: 0.17,
      duration: 100,
      yoyo: true,
      ease: 'Power2'
    });
    
    // Play jump sound
    this.playSound('jump', 400, 0.15);
  }
  
  private onJumpRelease() {
    // Variable jump - release early = shorter jump
    if (this.isJumping && this.player.body!.velocity.y < -200) {
      this.player.setVelocityY(-200); // Cap velocity for short jump
    }
  }

  private spawnObstaclePattern() {
    const { height } = this.cameras.main;
    const groundY = height - 80;
    
    const patterns = ['single', 'gap', 'lowhigh', 'cluster'];
    let pattern = Phaser.Math.RND.pick(patterns);
    
    // Don't repeat same pattern too often
    if (pattern === this.lastPattern && this.patternCounter < 2) {
      pattern = Phaser.Math.RND.pick(patterns.filter(p => p !== this.lastPattern));
    }
    
    if (pattern === this.lastPattern) {
      this.patternCounter++;
    } else {
      this.patternCounter = 0;
      this.lastPattern = pattern;
    }
    
    switch (pattern) {
      case 'single':
        this.spawnSingleObstacle(groundY);
        break;
      case 'gap':
        this.spawnGapPattern(groundY);
        break;
      case 'lowhigh':
        this.spawnLowHighPattern(groundY);
        break;
      case 'cluster':
        this.spawnClusterPattern(groundY);
        break;
    }
  }
  
  private spawnSingleObstacle(groundY: number) {
    const { width } = this.cameras.main;
    const obstacle = this.obstacles.create(width + 50, groundY - 40, 'obstacle') as Phaser.Physics.Arcade.Sprite;
    obstacle.setScale(0.15);
    obstacle.setVelocityX(-this.gameSpeed);
    obstacle.setImmovable(true);
    if (obstacle.body) {
      (obstacle.body as Phaser.Physics.Arcade.Body).setAllowGravity(false);
    }
  }
  
  private spawnGapPattern(groundY: number) {
    const { width } = this.cameras.main;
    // Two obstacles with gap in middle
    const obstacle1 = this.obstacles.create(width + 50, groundY - 40, 'obstacle') as Phaser.Physics.Arcade.Sprite;
    obstacle1.setScale(0.15);
    obstacle1.setVelocityX(-this.gameSpeed);
    obstacle1.setImmovable(true);
    if (obstacle1.body) {
      (obstacle1.body as Phaser.Physics.Arcade.Body).setAllowGravity(false);
    }
    
    const obstacle2 = this.obstacles.create(width + 250, groundY - 40, 'obstacle') as Phaser.Physics.Arcade.Sprite;
    obstacle2.setScale(0.15);
    obstacle2.setVelocityX(-this.gameSpeed);
    obstacle2.setImmovable(true);
    if (obstacle2.body) {
      (obstacle2.body as Phaser.Physics.Arcade.Body).setAllowGravity(false);
    }
  }
  
  private spawnLowHighPattern(groundY: number) {
    const { width } = this.cameras.main;
    // Low obstacle + high obstacle
    const low = this.obstacles.create(width + 50, groundY - 20, 'obstacle') as Phaser.Physics.Arcade.Sprite;
    low.setScale(0.12);
    low.setVelocityX(-this.gameSpeed);
    low.setImmovable(true);
    if (low.body) {
      (low.body as Phaser.Physics.Arcade.Body).setAllowGravity(false);
    }
    
    const high = this.obstacles.create(width + 200, groundY - 120, 'obstacle') as Phaser.Physics.Arcade.Sprite;
    high.setScale(0.15);
    high.setVelocityX(-this.gameSpeed);
    high.setImmovable(true);
    if (high.body) {
      (high.body as Phaser.Physics.Arcade.Body).setAllowGravity(false);
    }
  }
  
  private spawnClusterPattern(groundY: number) {
    const { width } = this.cameras.main;
    // Three obstacles close together
    for (let i = 0; i < 3; i++) {
      const obstacle = this.obstacles.create(width + 50 + (i * 80), groundY - 40, 'obstacle') as Phaser.Physics.Arcade.Sprite;
      obstacle.setScale(0.15);
      obstacle.setVelocityX(-this.gameSpeed);
      obstacle.setImmovable(true);
      if (obstacle.body) {
        (obstacle.body as Phaser.Physics.Arcade.Body).setAllowGravity(false);
      }
    }
  }

  private spawnStone() {
    const { width, height } = this.cameras.main;
    const groundY = height - 80;
    
    // Random stone type
    const stoneTypes = ['light', 'stone', 'metal', 'code', 'air', 'water', 'fire'];
    const randomType = Phaser.Math.RND.pick(stoneTypes);
    
    // Random height (ground or floating)
    const heightVariations = [
      groundY - 40,  // ground level
      groundY - 120, // mid air
      groundY - 200  // high air
    ];
    const y = Phaser.Math.RND.pick(heightVariations);
    
    const stone = this.stones.create(width + 50, y, `stone_${randomType}`) as Phaser.Physics.Arcade.Sprite;
    stone.setScale(0.6);
    stone.setVelocityX(-this.gameSpeed);
    if (stone.body) {
      (stone.body as Phaser.Physics.Arcade.Body).setAllowGravity(false);
    }
    
    // Add pulsing glow effect
    const timer = this.time.addEvent({
      delay: 50,
      callback: () => {
        const alpha = 0.5 + Math.sin(this.time.now * 0.01) * 0.5;
        stone.setAlpha(alpha);
      },
      loop: true
    });
    this.stoneGlowTimers.set(stone, timer);
  }

  private collectStone(_player: any, stone: any) {
    const x = stone.x;
    const y = stone.y;
    const currentTime = this.time.now;
    
    // Combo system
    if (currentTime - this.lastStoneTime < this.comboTimeout) {
      this.comboCount++;
      this.comboMultiplier = Math.min(1 + (this.comboCount * 0.5), 5); // Max 5x multiplier
      if (this.comboCount > this.bestCombo) {
        this.bestCombo = this.comboCount;
      }
    } else {
      this.comboCount = 1;
      this.comboMultiplier = 1.5;
    }
    this.lastStoneTime = currentTime;
    
    // Get stone type for color matching
    const stoneType = stone.texture.key.replace('stone_', '');
    const tribeColors: { [key: string]: number } = {
      light: 0xFFE066,
      stone: 0x8E9AAF,
      metal: 0xadb5bd,
      code: 0x0BB39C,
      air: 0x84C0C6,
      water: 0x4dabf7,
      fire: 0xF25F5C
    };
    const color = tribeColors[stoneType] || 0x0BB39C;
    
    // Create particle explosion - more particles for combo
    const particleCount = 15 + (this.comboCount * 5);
    const emitter = this.add.particles(x, y, 'particle', {
      speed: { min: 50, max: 150 },
      scale: { start: 0.4, end: 0 },
      tint: color,
      lifespan: 600,
      quantity: particleCount,
      blendMode: 'ADD',
      emitting: false
    });
    
    // Explode once and destroy after
    emitter.explode(particleCount);
    this.time.delayedCall(700, () => {
      emitter.destroy();
    });
    
    // Clean up glow timer
    const timer = this.stoneGlowTimers.get(stone);
    if (timer) {
      timer.destroy();
      this.stoneGlowTimers.delete(stone);
    }
    
    stone.destroy();
    this.stonesCollected++;
    this.stonesText.setText(`Stones: ${this.stonesCollected}`);
    
    // Combo display
    if (this.comboCount > 1) {
      this.comboText.setText(`${this.comboCount}x COMBO! ${this.comboMultiplier.toFixed(1)}x`);
      this.comboText.setColor('#FFE066');
      
      // Animate combo text
      this.tweens.add({
        targets: this.comboText,
        scaleX: 1.3,
        scaleY: 1.3,
        duration: 200,
        yoyo: true,
        ease: 'Power2'
      });
    } else {
      this.comboText.setText('');
    }
    
    // Animate score text
    this.tweens.add({
      targets: this.stonesText,
      scaleX: 1.2,
      scaleY: 1.2,
      duration: 150,
      yoyo: true,
      ease: 'Power2'
    });
    
    // Score with combo multiplier
    const baseScore = 50;
    const comboScore = Math.floor(baseScore * this.comboMultiplier);
    this.score += comboScore;
    
    // Play collect sound - higher pitch for combos
    const pitch = 400 + (this.comboCount * 50);
    this.playSound('collect', Math.min(pitch, 800), 0.2);
  }

  private hitObstacle() {
    this.isGameOver = true;
    this.physics.pause();
    
    // Reset combo on hit
    const bestComboDisplay = this.comboCount > 0 ? this.comboCount : this.bestCombo;
    this.comboCount = 0;
    this.comboMultiplier = 1;
    
    // Screen shake
    this.cameras.main.shake(300, 0.02);
    
    // Play hit sound
    this.playSound('hit', 150, 0.5);
    
    // Save high score
    if (this.score > this.highScore) {
      this.highScore = Math.floor(this.score);
      localStorage.setItem('reinmakerHighScore', this.highScore.toString());
    }
    
    // Game over screen
    const { width, height } = this.cameras.main;
    
    this.add.rectangle(width / 2, height / 2, 400, 300, 0x1B1E22, 0.9);
    
    const finalScore = Math.floor(this.score);
    const savedHighScore = parseInt(localStorage.getItem('reinmakerHighScore') || '0', 10);
    const isNewRecord = finalScore > savedHighScore;
    
    this.add.text(width / 2, height / 2 - 80, 'Game Over', {
      fontSize: '48px',
      color: '#F25F5C',
      fontStyle: 'bold'
    }).setOrigin(0.5);
    
    if (isNewRecord) {
      this.add.text(width / 2, height / 2 - 30, 'NEW HIGH SCORE!', {
        fontSize: '24px',
        color: '#FFE066',
        fontStyle: 'bold'
      }).setOrigin(0.5);
    }
    
    this.add.text(width / 2, height / 2 + 10, `Final Score: ${finalScore}`, {
      fontSize: '24px',
      color: '#F2F7FA'
    }).setOrigin(0.5);
    
    this.add.text(width / 2, height / 2 + 50, `Stones Collected: ${this.stonesCollected}`, {
      fontSize: '20px',
      color: '#0BB39C'
    }).setOrigin(0.5);
    
    if (bestComboDisplay > 0) {
      this.add.text(width / 2, height / 2 + 80, `Best Combo: ${bestComboDisplay}x`, {
        fontSize: '18px',
        color: '#FFE066'
      }).setOrigin(0.5);
    }
    
    this.add.text(width / 2, height / 2 + 120, 'Press R to Restart', {
      fontSize: '18px',
      color: '#adb5bd'
    }).setOrigin(0.5);
    
    // Play game over sound after a delay
    this.time.delayedCall(500, () => {
      this.playSound('gameOver', 200, 0.3);
    });
  }
  
  // Audio helper - generates musical tones using Web Audio API
  private playSound(type: 'jump' | 'collect' | 'hit' | 'gameOver', frequency: number, duration: number) {
    if (!this.audioContext) return;
    
    try {
      // Resume audio context if suspended
      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }
      
      const oscillator = this.audioContext.createOscillator();
      const gainNode = this.audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(this.audioContext.destination);
      
      // More musical wave types
      switch (type) {
        case 'jump':
          oscillator.type = 'sine';
          frequency = 440; // A4 note
          break;
        case 'collect':
          oscillator.type = 'triangle';
          // Play two notes for harmony
          this.playChord(frequency, frequency * 1.25, duration);
          return;
        case 'hit':
          oscillator.type = 'sawtooth';
          frequency = 150;
          break;
        case 'gameOver':
          oscillator.type = 'square';
          // Play descending notes
          this.playMelody([200, 180, 160, 140], duration * 4);
          return;
      }
      
      oscillator.frequency.value = frequency;
      gainNode.gain.setValueAtTime(0.2, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + duration);
    } catch (e) {
      console.debug('Audio playback failed:', e);
    }
  }
  
  private playChord(freq1: number, freq2: number, duration: number) {
    if (!this.audioContext) return;
    try {
      const osc1 = this.audioContext.createOscillator();
      const osc2 = this.audioContext.createOscillator();
      const gain = this.audioContext.createGain();
      
      osc1.type = 'triangle';
      osc2.type = 'triangle';
      osc1.frequency.value = freq1;
      osc2.frequency.value = freq2;
      
      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(this.audioContext.destination);
      
      gain.gain.setValueAtTime(0.15, this.audioContext.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);
      
      osc1.start();
      osc2.start();
      osc1.stop(this.audioContext.currentTime + duration);
      osc2.stop(this.audioContext.currentTime + duration);
    } catch (e) {
      // Fallback to single tone
      this.playSound('collect', freq1, duration);
    }
  }
  
  private playMelody(frequencies: number[], duration: number) {
    if (!this.audioContext) return;
    frequencies.forEach((freq, i) => {
      this.time.delayedCall(i * duration, () => {
        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();
        osc.connect(gain);
        gain.connect(this.audioContext.destination);
        osc.type = 'square';
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.2, this.audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);
        osc.start();
        osc.stop(this.audioContext.currentTime + duration);
      });
    });
  }
  
  // Pause menu
  private togglePause() {
    if (this.isGameOver) return;
    
    this.isPaused = !this.isPaused;
    
    if (this.isPaused) {
      this.physics.pause();
      this.showPauseMenu();
    } else {
      this.physics.resume();
      this.hidePauseMenu();
    }
  }
  
  private pauseMenuGroup!: Phaser.GameObjects.Container;
  
  private showPauseMenu() {
    const { width, height } = this.cameras.main;
    
    // Create container for pause menu elements
    this.pauseMenuGroup = this.add.container(width / 2, height / 2);
    
    // Background overlay
    const bg = this.add.rectangle(0, 0, width, height, 0x000000, 0.7);
    bg.setScrollFactor(0);
    
    // Menu box
    const menuBox = this.add.rectangle(0, 0, 300, 250, 0x1B1E22, 0.95);
    menuBox.setStrokeStyle(2, 0xD8A24A);
    
    // Title
    const title = this.add.text(0, -80, 'PAUSED', {
      fontSize: '36px',
      color: '#D8A24A',
      fontStyle: 'bold'
    }).setOrigin(0.5);
    
    // Resume button
    const resumeBtn = this.add.rectangle(0, -20, 200, 40, 0x0BB39C);
    const resumeText = this.add.text(0, -20, 'Resume (ESC)', {
      fontSize: '18px',
      color: '#FFFFFF'
    }).setOrigin(0.5);
    
    resumeBtn.setInteractive({ useHandCursor: true });
    resumeBtn.on('pointerdown', () => this.togglePause());
    
    // Restart button
    const restartBtn = this.add.rectangle(0, 30, 200, 40, 0xF25F5C);
    const restartText = this.add.text(0, 30, 'Restart (R)', {
      fontSize: '18px',
      color: '#FFFFFF'
    }).setOrigin(0.5);
    
    restartBtn.setInteractive({ useHandCursor: true });
    restartBtn.on('pointerdown', () => {
      this.scene.restart();
    });
    
    // Quit button
    const quitBtn = this.add.rectangle(0, 80, 200, 40, 0xadb5bd);
    const quitText = this.add.text(0, 80, 'Quit to Menu', {
      fontSize: '18px',
      color: '#FFFFFF'
    }).setOrigin(0.5);
    
    quitBtn.setInteractive({ useHandCursor: true });
    quitBtn.on('pointerdown', () => {
      this.scene.start('MenuScene');
    });
    
    // Add all to container
    this.pauseMenuGroup.add([bg, menuBox, title, resumeBtn, resumeText, restartBtn, restartText, quitBtn, quitText]);
  }
  
  private hidePauseMenu() {
    if (this.pauseMenuGroup) {
      this.pauseMenuGroup.destroy();
    }
  }
}
