import Phaser from 'phaser';

export default class MenuScene extends Phaser.Scene {
  constructor() {
    super({ key: 'MenuScene' });
  }

  preload() {
    // Load assets from public folder (served from root in Vite)
    // Core gameplay
    this.load.svg('player', 'player.svg', { width: 64, height: 64 });
    this.load.svg('obstacle', 'obstacle.svg', { width: 64, height: 64 });
    this.load.image('ground_stripe', 'ground_stripe.png'); // Keep or replace if needed
    
    // Background
    this.load.svg('bg', 'bg.svg', { width: 800, height: 600 });
    this.load.svg('ground_tex', 'ground_tex.svg', { width: 64, height: 64 });
    
    // Collectibles - Atoms (formerly Stones)
    this.load.svg('stone_light', 'stones/atom_light.svg', { width: 64, height: 64 });
    this.load.svg('stone_stone', 'stones/atom_stone.svg', { width: 64, height: 64 });
    this.load.svg('stone_metal', 'stones/atom_metal.svg', { width: 64, height: 64 });
    this.load.svg('stone_code', 'stones/atom_code.svg', { width: 64, height: 64 });
    this.load.svg('stone_air', 'stones/atom_air.svg', { width: 64, height: 64 });
    this.load.svg('stone_water', 'stones/atom_water.svg', { width: 64, height: 64 });
    this.load.svg('stone_fire', 'stones/atom_fire.svg', { width: 64, height: 64 });
    
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
      'Collect Knowledge Atoms',
      'Avoid Glitches',
      '',
      'Click to Start Mission'
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

