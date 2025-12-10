#!/usr/bin/env node
/**
 * Template Validation Script
 * Validates BASE_VIDEO_TEMPLATES.json against the schema
 * Run: node validate-templates.js
 */

const fs = require('fs');
const path = require('path');

const TEMPLATES_PATH = path.join(__dirname, 'BASE_VIDEO_TEMPLATES.json');
const SCHEMA_PATH = path.join(__dirname, 'base-video-templates-schema.json');

// ANSI colors for output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m'
};

function log(msg, color = 'reset') {
  console.log(`${colors[color]}${msg}${colors.reset}`);
}

function validateTemplates() {
  log('\n🎬 Kelly Base Video Templates Validator v1.0\n', 'cyan');
  log('─'.repeat(50), 'dim');

  // Load templates
  let templates;
  try {
    const raw = fs.readFileSync(TEMPLATES_PATH, 'utf-8');
    templates = JSON.parse(raw);
    log('✓ BASE_VIDEO_TEMPLATES.json loaded', 'green');
  } catch (err) {
    log(`✗ Failed to load templates: ${err.message}`, 'red');
    process.exit(1);
  }

  // Validation results
  const errors = [];
  const warnings = [];

  // 1. Check version
  if (!templates.version || !/^\d+\.\d+\.\d+$/.test(templates.version)) {
    errors.push('Missing or invalid version (must be semver)');
  } else {
    log(`✓ Version: ${templates.version}`, 'green');
  }

  // 2. Check global settings
  if (!templates.global) {
    errors.push('Missing global settings');
  } else {
    // Check sweater color
    const sweaterColor = templates.global.character?.sweater_color;
    if (sweaterColor !== '#B0C4DE') {
      errors.push(`Sweater color must be #B0C4DE, got: ${sweaterColor}`);
    } else {
      log('✓ Sweater color correct (#B0C4DE)', 'green');
    }

    // Check render settings
    const render = templates.global.render;
    if (render) {
      if (render.frame_rate < 60) {
        warnings.push(`Frame rate ${render.frame_rate}fps is below target 60fps`);
      } else {
        log(`✓ Frame rate: ${render.frame_rate}fps`, 'green');
      }

      const [width, height] = render.resolution || [];
      if (width < 1920 || height < 1080) {
        warnings.push(`Resolution ${width}x${height} is below minimum 1920x1080`);
      } else {
        log(`✓ Resolution: ${width}x${height}`, 'green');
      }
    }
  }

  // 3. Check animation layers
  if (!templates.animation_layers?.layers) {
    warnings.push('Missing animation layers definition');
  } else {
    const layers = templates.animation_layers.layers;
    const requiredLayers = ['breathing', 'idle_sway', 'blink', 'gesture', 'primary_expression'];
    const foundLayers = layers.map(l => l.name);
    
    for (const required of requiredLayers) {
      if (!foundLayers.includes(required)) {
        warnings.push(`Missing recommended layer: ${required}`);
      }
    }
    log(`✓ Animation layers: ${layers.length} defined`, 'green');
  }

  // 4. Check templates
  if (!templates.templates || !Array.isArray(templates.templates)) {
    errors.push('Missing templates array');
  } else {
    log(`\n📋 Validating ${templates.templates.length} templates...\n`, 'cyan');

    if (templates.templates.length < 8) {
      warnings.push(`Only ${templates.templates.length} templates (minimum 8 recommended)`);
    }

    for (const template of templates.templates) {
      const tid = template.id || 'UNKNOWN';
      const issues = [];

      // Required fields
      if (!template.id || !/^T\d{2}$/.test(template.id)) {
        issues.push('Invalid ID format (must be T01, T02, etc.)');
      }
      if (!template.name) issues.push('Missing name');
      if (!template.internal_name) issues.push('Missing internal_name');
      if (!template.category) issues.push('Missing category');
      if (!template.purpose) issues.push('Missing purpose');

      // Duration
      if (!template.duration) {
        issues.push('Missing duration settings');
      } else {
        if (template.duration.total_sec < 3 || template.duration.total_sec > 30) {
          issues.push(`Duration ${template.duration.total_sec}s outside valid range (3-30s)`);
        }
        if (template.duration.loop_end_sec <= template.duration.loop_start_sec) {
          issues.push('Loop end must be greater than loop start');
        }
      }

      // Camera
      if (!template.camera) {
        issues.push('Missing camera settings');
      } else {
        const validShots = ['wide', 'medium_full', 'medium', 'medium_close', 'close', 'extreme_close'];
        if (!validShots.includes(template.camera.shot_type)) {
          issues.push(`Invalid shot type: ${template.camera.shot_type}`);
        }
      }

      // Motion breakdown
      if (!template.motion_breakdown || Object.keys(template.motion_breakdown).length < 2) {
        issues.push('Motion breakdown must have at least 2 segments');
      }

      // Emotional arc
      if (!template.emotional_arc) {
        issues.push('Missing emotional arc');
      } else {
        const arc = template.emotional_arc;
        if (!arc.start || !arc.peak || !arc.end) {
          issues.push('Emotional arc must have start, peak, and end');
        }
      }

      // Prompt guidance
      if (!template.prompt_guidance?.minimax) {
        issues.push('Missing MiniMax prompt guidance');
      } else {
        const prompt = template.prompt_guidance.minimax;
        if (!prompt.includes('#B0C4DE') && !prompt.includes('powder blue')) {
          warnings.push(`${tid}: Prompt may be missing sweater color reference`);
        }
        if (prompt.length < 100) {
          warnings.push(`${tid}: Prompt seems too short (${prompt.length} chars)`);
        }
      }

      // Report template status
      if (issues.length === 0) {
        log(`  ✓ ${tid}: ${template.name}`, 'green');
      } else {
        log(`  ✗ ${tid}: ${template.name}`, 'red');
        for (const issue of issues) {
          errors.push(`${tid}: ${issue}`);
          log(`      - ${issue}`, 'red');
        }
      }
    }
  }

  // 5. Check phase mapping
  if (!templates.phase_to_template_mapping) {
    warnings.push('Missing phase_to_template_mapping');
  } else {
    const phases = Object.keys(templates.phase_to_template_mapping);
    const requiredPhases = ['welcome', 'hook', 'q1', 'q2', 'q3', 'wisdom', 'closing'];
    for (const phase of requiredPhases) {
      if (!phases.includes(phase)) {
        warnings.push(`Missing phase mapping: ${phase}`);
      }
    }
    log(`\n✓ Phase mappings: ${phases.length} defined`, 'green');
  }

  // Summary
  log('\n' + '─'.repeat(50), 'dim');
  log('\n📊 VALIDATION SUMMARY\n', 'cyan');

  if (errors.length === 0 && warnings.length === 0) {
    log('🎉 All validations passed!', 'green');
  } else {
    if (errors.length > 0) {
      log(`❌ Errors: ${errors.length}`, 'red');
      for (const err of errors) {
        log(`   • ${err}`, 'red');
      }
    }
    if (warnings.length > 0) {
      log(`⚠️  Warnings: ${warnings.length}`, 'yellow');
      for (const warn of warnings) {
        log(`   • ${warn}`, 'yellow');
      }
    }
  }

  // Template coverage report
  log('\n📈 TEMPLATE COVERAGE\n', 'cyan');
  
  const categories = {};
  for (const t of templates.templates || []) {
    const cat = t.category || 'unknown';
    categories[cat] = (categories[cat] || 0) + 1;
  }
  
  for (const [cat, count] of Object.entries(categories)) {
    log(`  ${cat}: ${count} template(s)`, 'dim');
  }

  // Phase coverage
  log('\n📍 PHASE COVERAGE\n', 'cyan');
  const phaseMap = templates.phase_to_template_mapping || {};
  for (const [phase, tids] of Object.entries(phaseMap)) {
    const coverage = tids.length >= 2 ? '✓' : '⚠';
    const color = tids.length >= 2 ? 'green' : 'yellow';
    log(`  ${coverage} ${phase}: ${tids.join(', ')}`, color);
  }

  log('\n' + '─'.repeat(50), 'dim');
  log('');

  // Exit code
  process.exit(errors.length > 0 ? 1 : 0);
}

// Run validation
validateTemplates();



