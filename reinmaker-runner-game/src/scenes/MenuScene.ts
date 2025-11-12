import Phaser from 'phaser';

export default class MenuScene extends Phaser.Scene {
  constructor() {
    super({ key: 'MenuScene' });
  }

  preload() {
    // Load assets from public folder (served from root in Vite)
    // Core gameplay
    this.load.image('player', 'player.png');
    this.load.image('obstacle', 'obstacle.png');
    this.load.image('ground_stripe', 'ground_stripe.png');
    
    // Background
    this.load.image('bg', 'bg.png');
    this.load.image('ground_tex', 'ground_tex.png');
    
    // Collectibles - Knowledge Stones
    this.load.image('stone_light', 'stones/stone_light.png');
    this.load.image('stone_stone', 'stones/stone_stone.png');
    this.load.image('stone_metal', 'stones/stone_metal.png');
    this.load.image('stone_code', 'stones/stone_code.png');
    this.load.image('stone_air', 'stones/stone_air.png');
    this.load.image('stone_water', 'stones/stone_water.png');
    this.load.image('stone_fire', 'stones/stone_fire.png');
    
    // UI
    this.load.image('favicon', 'favicon.png');
  }

  create() {
    const { width, height } = this.cameras.main;
    
    // Title
    this.add.text(width / 2, height / 3, 'The Rein Maker\'s Daughter', {
      fontSize: '36px',
      color: '#D8A24A',
      fontStyle: 'bold'
    }).setOrigin(0.5);
    
    // Subtitle
    this.add.text(width / 2, height / 3 + 50, 'A Runner Game', {
      fontSize: '18px',
      color: '#adb5bd'
    }).setOrigin(0.5);
    
    // Instructions
    const instructions = [
      'SPACE or UP ARROW to jump',
      'Collect Knowledge Stones',
      'Avoid obstacles',
      '',
      'Click to Start'
    ];
    
    this.add.text(width / 2, height / 2 + 40, instructions.join('\n'), {
      fontSize: '16px',
      color: '#F2F7FA',
      align: 'center',
      lineSpacing: 8
    }).setOrigin(0.5);
    
    // Make the scene clickable
    this.input.on('pointerdown', () => {
      this.scene.start('GameScene');
    });
    
    // Also allow spacebar to start
    this.input.keyboard?.once('keydown-SPACE', () => {
      this.scene.start('GameScene');
    });
  }
}

