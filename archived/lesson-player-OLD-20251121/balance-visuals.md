# Balance Lesson - Visual Assets Manifest

## Core Visual Requirements

### Kelly Avatar Assets
**Location:** `content/assets/models/kelly/`

- `kelly_age_3.glb` - Toddler Kelly (playful, wobbly)
- `kelly_age_9.glb` - Child Kelly (curious, energetic)
- `kelly_age_15.glb` - Teen Kelly (cool, relatable)
- `kelly_age_27.glb` - Adult Kelly (professional, warm)
- `kelly_age_48.glb` - Middle-age Kelly (wise, experienced)
- `kelly_age_82.glb` - Elder Kelly (graceful, profound)

### Animation Sets
**Location:** `content/assets/animations/balance/`

#### Universal Animations (All Ages)
- `balance_idle.anim` - Standing neutral
- `balance_wobble_light.anim` - Small balance adjustments
- `balance_wobble_medium.anim` - Noticeable wobbling
- `balance_wobble_extreme.anim` - Almost falling
- `balance_recover.anim` - Catching balance
- `balance_fall.anim` - Falling animation (playful)
- `balance_one_foot.anim` - Standing on one foot
- `balance_arms_out.anim` - Arms extended for balance

#### Age-Specific Animations
**Ages 2-5:**
- `toddler_giggle_wobble.anim` - Giggling while wobbling
- `toddler_proud_pose.anim` - "Look at me!" pose
- `toddler_plop_sit.anim` - Falling to sitting (cute)

**Ages 6-12:**
- `child_bike_balance.anim` - Pretending to ride bike
- `child_tightrope_walk.anim` - Arms out, careful steps
- `child_spin_dizzy.anim` - Spinning then wobbling

**Ages 13-17:**
- `teen_phone_walk.anim` - Walking while texting
- `teen_skateboard_balance.anim` - Skateboard stance
- `teen_multitask_juggle.anim` - Juggling gesture

**Ages 18-35:**
- `adult_yoga_tree_pose.anim` - Professional balance pose
- `adult_coffee_balance.anim` - Balancing coffee and laptop
- `adult_mindful_breathing.anim` - Centering breath

**Ages 36-60:**
- `midlife_careful_balance.anim` - More cautious movement
- `midlife_helping_gesture.anim` - Supporting someone
- `midlife_contemplative_nod.anim` - Wise understanding

**Ages 61-102:**
- `elder_tai_chi.anim` - Slow, graceful movements
- `elder_walking_stick.anim` - Using support wisely
- `elder_sage_gesture.anim` - Profound hand movements

### Concept Visualizations
**Location:** `content/assets/diagrams/balance/`

#### Physics & Body
- `vestibular_system_simple.svg` - Ages 2-5 (cartoon ear with smiling crystals)
- `vestibular_system_detailed.svg` - Ages 6-12 (labeled anatomy)
- `vestibular_system_scientific.svg` - Ages 13+ (medical accuracy)
- `center_of_gravity_examples.svg` - Progressive complexity by age
- `force_vectors_seesaw.svg` - Physics of balance
- `torque_demonstration.svg` - Lever arms and rotation

#### Mathematical Balance
- `balance_scale_equal.svg` - Simple equality (2-5)
- `equation_balance_basic.svg` - 3 + 4 = 7 visual (6-12)
- `equation_balance_algebra.svg` - 2x + 3 = 11 solving (13-17)
- `optimization_graphs.svg` - Calculus balance points (18-35)
- `symmetry_examples.svg` - Geometric balance
- `normal_distribution.svg` - Statistical balance

#### Life Balance
- `sharing_toys_equal.svg` - Fair sharing visual (2-5)
- `daily_schedule_pie.svg` - Time balance (6-12)
- `life_balance_wheel.svg` - 8 life domains (13-17)
- `work_life_integration.svg` - Modern balance (18-35)
- `sandwich_generation.svg` - Multiple responsibilities (36-60)
- `life_seasons_mandala.svg` - Lifetime perspective (61-102)

#### Nature & Systems
- `tree_roots_branches.svg` - Natural balance (all ages, complexity varies)
- `ecosystem_food_web.svg` - Balanced relationships
- `water_cycle_balance.svg` - Natural equilibrium
- `climate_system_balance.svg` - Global systems

### Interactive Elements
**Location:** `content/assets/interactive/balance/`

- `balance_beam_game.html` - Tilt to balance game
- `equation_balance_interactive.html` - Drag numbers to balance
- `life_wheel_assessment.html` - Rate your balance areas
- `wobble_meter.html` - Shows balance/imbalance
- `seesaw_simulator.html` - Add/remove weights

### Background Environments
**Location:** `content/assets/backgrounds/balance/`

**Ages 2-5:** 
- `playground_seesaw.jpg` - Bright, colorful playground
- `bedroom_blocks.jpg` - Cozy room with toys

**Ages 6-12:**
- `gymnasium.jpg` - School gym with balance beam
- `science_classroom.jpg` - Classroom with experiments

**Ages 13-17:**
- `skate_park.jpg` - Urban balance environment
- `teen_bedroom.jpg` - Relatable personal space

**Ages 18-35:**
- `modern_office.jpg` - Work-life balance setting
- `yoga_studio.jpg` - Wellness space

**Ages 36-60:**
- `home_kitchen.jpg` - Family hub
- `garden_path.jpg` - Peaceful natural setting

**Ages 61-102:**
- `park_bench.jpg` - Contemplative outdoor space
- `library_wisdom.jpg` - Knowledge and reflection

### Supporting Icons & UI
**Location:** `content/assets/ui/balance/`

- `wobble_indicator.svg` - Shows degree of wobble
- `balance_meter.svg` - Visual balance indicator
- `choice_buttons_balance.svg` - Themed choice buttons
- `progress_bar_balance.svg` - Lesson progress tracker
- `celebration_stars.svg` - Success feedback
- `encouragement_hearts.svg` - Supportive feedback

### Audio-Visual Sync Markers
**Location:** `content/assets/sync/balance/`

- `lip_sync_phonemes.json` - Phoneme to viseme mappings
- `emphasis_markers.json` - Word emphasis visual cues
- `gesture_timing.json` - Animation trigger points
- `expression_timeline.json` - Facial expression changes

## Asset Production Guidelines

### Technical Specifications
- **3D Models:** GLTF 2.0 format, <5MB per model
- **Animations:** 30fps, loopable where appropriate
- **Images:** WebP format, responsive sizes (360p, 720p, 1080p)
- **SVGs:** Optimized, accessible with ARIA labels
- **Interactive:** HTML5, touch-friendly, keyboard accessible

### Accessibility Requirements
- Alt text for all images
- High contrast versions available
- Reduced motion alternatives
- Screen reader descriptions
- Colorblind-friendly palettes

### Cultural Considerations
- Diverse representation in human figures
- Universal symbols where possible
- Avoid culture-specific gestures
- Multiple skin tone options for hands/avatars

### Quality Checklist
- [ ] Age-appropriate complexity
- [ ] Scientifically accurate
- [ ] Emotionally supportive
- [ ] Engaging but not distracting
- [ ] Loads quickly (<3 seconds)
- [ ] Works offline after initial load
- [ ] Graceful fallbacks for missing assets

## Priority Order for Production

### Must Have (MVP)
1. Kelly age 3, 9, 27 models
2. Basic wobble animations
3. Simple balance diagrams
4. Core backgrounds

### Should Have (Enhanced)
1. All 6 Kelly age models
2. Full animation library
3. Interactive elements
4. Mathematical visualizations

### Nice to Have (Deluxe)
1. Seasonal variations
2. Cultural adaptations
3. Advanced physics simulations
4. AR/VR ready assets

## File Naming Conventions

```
{concept}_{agegroup}_{type}_{variation}.{ext}
```

Examples:
- `vestibular_2-5_diagram_simple.svg`
- `kelly_age_9_model_v2.glb`
- `wobble_all_animation_medium.anim`
- `balance_13-17_interactive_game.html`

## Asset Dependencies

```yaml
balance_lesson:
  requires:
    - kelly_avatar: [age_appropriate_model]
    - animations: [wobble, balance, recover]
    - diagrams: [core_concept_visual]
    - background: [age_appropriate_scene]
  optional:
    - interactive: [balance_game]
    - advanced: [physics_simulations]
```

## Version Control

All assets tracked in:
```
content/assets/
├── manifest.json  # Asset registry with versions
├── balance/       # Lesson-specific assets
│   ├── v1/       # Current production
│   └── v2/       # In development
└── shared/        # Cross-lesson assets
```
