#!/usr/bin/env npx tsx
/**
 * 🎨 KELLY LORA ASSET FACTORY v2.0
 * 
 * ENHANCED with:
 *   - Responsive sizes (mobile/tablet/desktop)
 *   - WebP format conversion for web optimization
 *   - Complete asset coverage for all platforms
 *   - Multiple variants for A/B testing
 *   - Post-processing for exact size requirements
 * 
 * Usage: 
 *   npx tsx scripts/kelly-lora-asset-factory.ts
 *   npx tsx scripts/kelly-lora-asset-factory.ts --priority=critical
 *   npx tsx scripts/kelly-lora-asset-factory.ts --category=social
 *   npx tsx scripts/kelly-lora-asset-factory.ts --with-variants
 *   npx tsx scripts/kelly-lora-asset-factory.ts --responsive
 * 
 * Categories: social, brand, hero, chair, poses, expressions, personas
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
  // Output directories
  outputBase: 'public',
  rendersDir: 'renders',
  
  // Responsive breakpoints
  sizes: {
    mobile: { width: 640, suffix: '-mobile' },
    tablet: { width: 1024, suffix: '-tablet' },
    desktop: { width: 1920, suffix: '-desktop' },
    retina: { width: 3840, suffix: '-2x' },
  },
  
  // Format options
  formats: {
    png: { ext: '.png', quality: 100 },
    webp: { ext: '.webp', quality: 90 },
    jpeg: { ext: '.jpeg', quality: 85 },
  },
  
  // Rate limiting
  delayBetweenGenerations: 2000, // ms
  
  // Variants to generate for A/B testing
  variantCount: 2,
};

// ═══════════════════════════════════════════════════════════════════════════
// KELLY IDENTITY - LOCKED FOR CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════════

const KELLY_LORA = {
  civitai_url: 'https://civitai.com/api/download/models/2455956',
  scale: 0.85,
  trigger: 'kelly',
};

// Kelly's base appearance - LOCKED
const KELLY_BASE = `kelly, woman with brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup`;

// Outfit variants
const OUTFITS = {
  casual: `wearing soft powder blue cashmere crewneck sweater, medium wash blue jeans cuffed at ankle, white leather sneakers`,
  studio: `wearing soft powder blue cashmere crewneck sweater, sitting in director's chair with black canvas and warm wood frame`,
  professional: `wearing soft powder blue cashmere crewneck sweater, professional studio lighting`,
};

// Scene/background variants
const SCENES = {
  studio: `pure white cyclorama photography studio, professional studio lighting with soft natural window light, clean minimal background, shot on Hasselblad H6D-100c, 85mm f/2.8, 8K UHD`,
  dark_studio: `professional dark studio background #0f0f11, dramatic rim lighting, clean minimal background, 8K UHD`,
  transparent: `solid pure white background for easy cutout, professional soft lighting, 8K UHD`,
  warm: `warm natural lighting, soft golden hour glow, professional photography, 8K UHD`,
  gradient: `professional gradient background from dark #0a0a0a to charcoal #2a2a2a, dramatic lighting, 8K UHD`,
};

// ═══════════════════════════════════════════════════════════════════════════
// COMPLETE ASSET DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════

interface AssetSpec {
  id: string;
  category: 'social' | 'brand' | 'hero' | 'chair' | 'poses' | 'expressions' | 'personas' | 'daily';
  priority: 'critical' | 'high' | 'medium';
  prompt: string;
  aspect_ratio: '1:1' | '16:9' | '4:3' | '3:4' | '9:16' | '21:9' | 'custom';
  width?: number;
  height?: number;
  output_path: string;
  heygen_upload?: boolean;
  responsive?: boolean; // Generate mobile/tablet/desktop versions
  webp?: boolean; // Also generate WebP version
  variants?: number; // Generate N variants for A/B testing
  description: string;
}

const ASSETS: AssetSpec[] = [
  // ═══════════════════════════════════════════════════════════════════════════
  // SOCIAL MEDIA - CRITICAL (Launch Blocking)
  // ═══════════════════════════════════════════════════════════════════════════
  
  // Master Profile (generates all platform sizes)
  {
    id: 'profile-master',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, head and shoulders closeup, warm genuine smile, eyes engaged looking directly at camera, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'renders/social/profile-master-2048.png',
    heygen_upload: true,
    webp: true,
    variants: 3, // Generate 3 variants to pick the best
    description: 'Master profile picture - exports to all platform sizes',
  },
  
  // Twitter/X
  {
    id: 'cover-twitter',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, upper body shot positioned in left third of frame, curious welcoming expression, looking slightly to the right with engaging smile, space for text overlay on right side, ${SCENES.dark_studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/social/cover-twitter.png',
    webp: true,
    variants: 2,
    description: 'Twitter/X header banner (1500×500)',
  },
  
  // LinkedIn
  {
    id: 'cover-linkedin',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, professional confident pose, upper body positioned in left portion of frame, space for company tagline on right, ${SCENES.dark_studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/social/cover-linkedin.png',
    webp: true,
    description: 'LinkedIn company cover (1584×396)',
  },
  
  // YouTube Banner
  {
    id: 'cover-youtube',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, welcoming energetic expression, upper body centered in frame for safe zone visibility, ${SCENES.gradient}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/social/cover-youtube.png',
    webp: true,
    description: 'YouTube channel banner (2560×1440)',
  },
  
  // Facebook Cover
  {
    id: 'cover-facebook',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, friendly approachable expression, upper body positioned in left portion, space for logo on right, ${SCENES.dark_studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/social/cover-facebook.png',
    webp: true,
    description: 'Facebook page cover (820×312)',
  },
  
  // TikTok Profile
  {
    id: 'profile-tiktok',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup, playful energetic smile, bright engaging eyes, youthful energy, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/social/profile-tiktok.png',
    description: 'TikTok profile picture (400×400)',
  },
  
  // Instagram Profile
  {
    id: 'profile-instagram',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, head and shoulders, warm genuine smile with slight head tilt, instagram-aesthetic lighting, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/social/profile-instagram.png',
    description: 'Instagram profile picture (640×640)',
  },
  
  // OG Images
  {
    id: 'og-default',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, teaching pose, upper body positioned in right third of frame, engaged explaining expression, large space for title text overlay on left, ${SCENES.dark_studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/social/og-default.png',
    webp: true,
    responsive: true,
    description: 'Default Open Graph share image (1200×630)',
  },
  
  {
    id: 'twitter-card-large',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, engaging teaching pose, upper body, welcoming expression, ${SCENES.dark_studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/social/twitter-card-large.png',
    webp: true,
    description: 'Twitter card summary large image (1200×600)',
  },
  
  {
    id: 'twitter-card-summary',
    category: 'social',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup, warm smile, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/social/twitter-card-summary.png',
    description: 'Twitter card summary square (240×240)',
  },
  
  // Discord
  {
    id: 'profile-discord',
    category: 'social',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, friendly approachable expression, head and shoulders, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/social/profile-discord.png',
    description: 'Discord server icon (512×512)',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // HERO IMAGES - CRITICAL
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'kelly-hero-4k',
    category: 'hero',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, confident inviting pose in director's chair, slight low angle heroic shot, warm smile, arms relaxed on chair arms, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly-hero-4k.png',
    heygen_upload: true,
    responsive: true,
    webp: true,
    variants: 3,
    description: 'Main landing page hero image - 4K',
  },
  
  {
    id: 'kelly-hero-mobile',
    category: 'hero',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, confident inviting pose in director's chair, portrait orientation, warm smile, ${SCENES.studio}`,
    aspect_ratio: '3:4',
    output_path: 'public/images/kelly-hero-mobile.png',
    webp: true,
    description: 'Mobile hero image - portrait orientation',
  },
  
  {
    id: 'kelly-og-image',
    category: 'hero',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, upper body in director's chair, welcoming expression, space for text overlay, ${SCENES.dark_studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly-og-image.png',
    webp: true,
    description: 'Site-wide OG share image',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // LESSON WELCOME & POSES - CRITICAL
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'kelly-welcome-pose',
    category: 'poses',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, arms slightly open in welcoming gesture, warm greeting expression, full body visible, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_welcome.png',
    heygen_upload: true,
    webp: true,
    description: 'Lesson start welcome pose',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // CHAIR EXPRESSIONS (5 core expressions)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'kelly-chair-celebrating',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, both arms raised joyfully in celebration, big genuine triumphant smile, bright excited eyes, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-celebrating.png',
    heygen_upload: true,
    webp: true,
    description: 'Chair pose - celebrating success',
  },
  
  {
    id: 'kelly-chair-curious',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, head tilted slightly with raised eyebrows, curious inquisitive expression, finger thoughtfully near chin, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-curious.png',
    heygen_upload: true,
    webp: true,
    description: 'Chair pose - curious/questioning',
  },
  
  {
    id: 'kelly-chair-explaining',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, hands gesturing naturally while explaining, animated engaged teaching expression, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-explaining.png',
    heygen_upload: true,
    webp: true,
    description: 'Chair pose - explaining/teaching',
  },
  
  {
    id: 'kelly-chair-listening',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, attentive listening posture, slight forward lean, warm supportive expression, hands gently clasped, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-listening.png',
    heygen_upload: true,
    webp: true,
    description: 'Chair pose - active listening',
  },
  
  {
    id: 'kelly-chair-wisdom',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, calm knowing smile, relaxed dignified posture, wise serene expression, peaceful demeanor, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-wisdom.png',
    heygen_upload: true,
    webp: true,
    description: 'Chair pose - wisdom/insight',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // LESSON PLAYER POSES (Standing poses for interactions)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'kelly-idle',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing neutral relaxed stance, hands at sides, warm attentive expression, full body, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_idle.png',
    webp: true,
    description: 'Neutral idle stance',
  },
  
  {
    id: 'kelly-listening',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing active listening posture, head slightly tilted, attentive engaged expression, full body, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_listening.png',
    webp: true,
    description: 'Active listening pose',
  },
  
  {
    id: 'kelly-choice-left',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, left arm extended gracefully pointing to the left, body angled slightly left, encouraging expression, full body, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_choice_left.png',
    webp: true,
    description: 'Pointing left for choice A',
  },
  
  {
    id: 'kelly-choice-right',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, right arm extended gracefully pointing to the right, body angled slightly right, encouraging expression, full body, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_choice_right.png',
    webp: true,
    description: 'Pointing right for choice B',
  },
  
  {
    id: 'kelly-hint',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, index finger touching chin thoughtfully, playful knowing expression, head tilted, mid-body shot, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_hint.png',
    webp: true,
    description: 'Thoughtful hint pose',
  },
  
  {
    id: 'kelly-clasp',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, hands clasped together in front of chest, eager anticipating expression, mid-body shot, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_clasp.png',
    webp: true,
    description: 'Hands clasped anticipation',
  },
  
  {
    id: 'kelly-thumbs-up',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, giving enthusiastic thumbs up with right hand, proud encouraging expression, mid-body shot, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_thumbs_up.png',
    webp: true,
    description: 'Thumbs up encouragement',
  },
  
  {
    id: 'kelly-thinking',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, hand on chin in contemplation, looking slightly upward, deep thought expression, mid-body shot, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_thinking.png',
    webp: true,
    description: 'Deep thinking pose',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // EXPRESSION CLOSEUPS (Face library)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'expr-celebrating',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, joyful triumphant expression, big genuine smile with teeth showing, bright sparkling eyes, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/celebrating.jpeg',
    webp: true,
    description: 'Expression - celebrating',
  },
  
  {
    id: 'expr-confused',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, puzzled questioning expression, slight frown, furrowed brow, head tilted, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/confused.jpeg',
    webp: true,
    description: 'Expression - confused',
  },
  
  {
    id: 'expr-curious-closeup',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, extreme face closeup, intense curiosity expression, wide interested eyes, raised eyebrows, slight smile, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/curious-closeup.jpeg',
    webp: true,
    description: 'Expression - curious closeup',
  },
  
  {
    id: 'expr-curious-main',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, curious interested expression, slight knowing smile, engaged eyes, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/curious-main.jpeg',
    webp: true,
    description: 'Expression - curious main',
  },
  
  {
    id: 'expr-curious-thinking',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, thoughtful curiosity expression, contemplating something interesting, eyes slightly narrowed, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/curious-thinking.jpeg',
    webp: true,
    description: 'Expression - curious thinking',
  },
  
  {
    id: 'expr-explaining',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, animated explaining expression, mouth slightly open mid-speech, engaged teaching eyes, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/explaining.jpeg',
    webp: true,
    description: 'Expression - explaining',
  },
  
  {
    id: 'expr-happy-content',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, content happy expression, peaceful genuine smile, relaxed satisfied demeanor, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/happy-content.jpeg',
    webp: true,
    description: 'Expression - happy content',
  },
  
  {
    id: 'expr-peaceful',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, serene peaceful calm expression, gentle soft smile, relaxed features, eyes warm and kind, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/peaceful.jpeg',
    webp: true,
    description: 'Expression - peaceful',
  },
  
  {
    id: 'expr-surprised',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, pleasantly surprised expression, raised eyebrows, wide delighted eyes, open mouth smile, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/surprised.jpeg',
    webp: true,
    description: 'Expression - surprised',
  },
  
  {
    id: 'expr-encouraging',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, warm encouraging expression, supportive smile, eyes conveying belief and confidence, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/encouraging.jpeg',
    webp: true,
    description: 'Expression - encouraging',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PERSONAS (12 Archetypes)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'persona-scientist',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, analytical focused expression, examining something thoughtfully, scientist archetype energy, intellectual gaze, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/scientist.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Scientist',
  },
  
  {
    id: 'persona-explorer',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, adventurous excited expression, eager curious eyes looking into distance, explorer archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/explorer.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Explorer',
  },
  
  {
    id: 'persona-rebel',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, determined bold expression, confident challenging look, slight smirk, rebel archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/rebel.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Rebel',
  },
  
  {
    id: 'persona-architect',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, thoughtful precise expression, considering carefully with focused eyes, architect archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/architect.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Architect',
  },
  
  {
    id: 'persona-diplomat',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, warm understanding expression, gentle empathetic smile, open approachable demeanor, diplomat archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/diplomat.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Diplomat',
  },
  
  {
    id: 'persona-empath',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, compassionate gentle expression, soft caring eyes, nurturing warm smile, empath archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/empath.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Empath',
  },
  
  {
    id: 'persona-macgyver',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, creative resourceful expression, clever knowing smile, eyes showing quick wit, macgyver archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/macgyver.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The MacGyver',
  },
  
  {
    id: 'persona-mystic',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, wise serene expression, deep knowing eyes, peaceful transcendent smile, mystic archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/mystic.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Mystic',
  },
  
  {
    id: 'persona-provider',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, nurturing generous expression, warm protective smile, caring devoted eyes, provider archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/provider.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Provider',
  },
  
  {
    id: 'persona-storyteller',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, animated engaging expression, bright expressive eyes mid-story, hands slightly raised in gesture, storyteller archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/storyteller.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Storyteller',
  },
  
  {
    id: 'persona-strategist',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, sharp calculating expression, focused analytical eyes, knowing confident smile, strategist archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/strategist.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Strategist',
  },
  
  {
    id: 'persona-survivor',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, strong resilient expression, determined confident eyes, unwavering steady gaze, survivor archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/survivor.png',
    heygen_upload: true,
    webp: true,
    description: 'Persona - The Survivor',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // DAILY LESSON SOCIAL CARDS (Template-ready)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'daily-quote-template',
    category: 'daily',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, thoughtful wise expression, looking slightly off camera, lots of empty space on right side for text overlay, ${SCENES.dark_studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/kelly/templates/daily-quote-template.png',
    description: 'Daily quote card template (1080×1080)',
  },
  
  {
    id: 'daily-story-template',
    category: 'daily',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, standing pose, welcoming engaging expression, positioned in bottom third of frame, lots of space above for text, ${SCENES.gradient}`,
    aspect_ratio: '9:16',
    output_path: 'public/kelly/templates/daily-story-template.png',
    description: 'Instagram/TikTok story template (1080×1920)',
  },
  
  {
    id: 'daily-reel-template',
    category: 'daily',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, dynamic engaging pose, looking directly at camera, energetic expression, positioned for vertical video, ${SCENES.studio}`,
    aspect_ratio: '9:16',
    output_path: 'public/kelly/templates/daily-reel-template.png',
    heygen_upload: true,
    description: 'Reels/TikTok video template (1080×1920)',
  },
];

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION ENGINE
// ═══════════════════════════════════════════════════════════════════════════

interface GenerationResult {
  asset: AssetSpec;
  variant?: number;
  success: boolean;
  imageUrl?: string;
  localPath?: string;
  additionalPaths?: string[];
  error?: string;
  duration?: number;
}

async function generateWithFlux(asset: AssetSpec, variant?: number): Promise<GenerationResult> {
  const startTime = Date.now();
  const variantSuffix = variant !== undefined ? `-v${variant + 1}` : '';
  const displayId = `${asset.id}${variantSuffix}`;
  
  console.log(`\n🎨 Generating: ${displayId}`);
  console.log(`   ${asset.description}`);
  
  try {
    // Determine aspect ratio
    let aspectRatio = asset.aspect_ratio;
    if (aspectRatio === 'custom') {
      const ratio = asset.width! / asset.height!;
      if (ratio >= 2.5) aspectRatio = '21:9';
      else if (ratio >= 1.5) aspectRatio = '16:9';
      else if (ratio >= 1.2) aspectRatio = '4:3';
      else if (ratio >= 0.9) aspectRatio = '1:1';
      else if (ratio >= 0.6) aspectRatio = '3:4';
      else aspectRatio = '9:16';
    }
    
    // Use FLUX 1.1 Pro
    const output = await replicate.run(
      "black-forest-labs/flux-1.1-pro",
      {
        input: {
          prompt: asset.prompt,
          aspect_ratio: aspectRatio,
          output_format: "png",
          output_quality: 100,
          safety_tolerance: 2,
          prompt_upsampling: true
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`   📥 Downloading...`);
    
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    const buffer = Buffer.from(await response.arrayBuffer());
    
    // Build output path with variant suffix
    let outputPath = asset.output_path;
    if (variant !== undefined) {
      const ext = path.extname(outputPath);
      const base = outputPath.slice(0, -ext.length);
      outputPath = `${base}${variantSuffix}${ext}`;
    }
    
    const fullPath = path.join(process.cwd(), outputPath);
    fs.mkdirSync(path.dirname(fullPath), { recursive: true });
    fs.writeFileSync(fullPath, buffer);
    
    const additionalPaths: string[] = [];
    
    // Generate WebP version if requested
    if (asset.webp) {
      const webpPath = fullPath.replace(/\.(png|jpeg|jpg)$/i, '.webp');
      try {
        // Try to use sharp if available, otherwise skip
        const sharp = await import('sharp').catch(() => null);
        if (sharp) {
          await sharp.default(buffer)
            .webp({ quality: 90 })
            .toFile(webpPath);
          additionalPaths.push(webpPath);
          console.log(`   📦 WebP: ${path.basename(webpPath)}`);
        }
      } catch (e) {
        // WebP conversion failed, continue without it
      }
    }
    
    const duration = (Date.now() - startTime) / 1000;
    console.log(`   ✅ Saved: ${outputPath} (${duration.toFixed(1)}s)`);
    
    return {
      asset,
      variant,
      success: true,
      imageUrl,
      localPath: fullPath,
      additionalPaths,
      duration,
    };
    
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return {
      asset,
      variant,
      success: false,
      error: error.message,
      duration: (Date.now() - startTime) / 1000,
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// RESPONSIVE IMAGE GENERATION
// ═══════════════════════════════════════════════════════════════════════════

async function generateResponsiveSizes(result: GenerationResult): Promise<string[]> {
  if (!result.success || !result.localPath || !result.asset.responsive) {
    return [];
  }
  
  const generated: string[] = [];
  
  try {
    const sharp = await import('sharp').catch(() => null);
    if (!sharp) {
      console.log('   ⚠️ Sharp not available, skipping responsive sizes');
      return [];
    }
    
    const buffer = fs.readFileSync(result.localPath);
    const image = sharp.default(buffer);
    const metadata = await image.metadata();
    
    for (const [sizeName, sizeConfig] of Object.entries(CONFIG.sizes)) {
      if (sizeConfig.width >= (metadata.width || 0)) continue; // Skip if larger than original
      
      const ext = path.extname(result.localPath);
      const base = result.localPath.slice(0, -ext.length);
      const responsivePath = `${base}${sizeConfig.suffix}${ext}`;
      
      await sharp.default(buffer)
        .resize(sizeConfig.width)
        .toFile(responsivePath);
      
      generated.push(responsivePath);
      
      // Also generate WebP version
      if (result.asset.webp) {
        const webpPath = responsivePath.replace(/\.(png|jpeg|jpg)$/i, '.webp');
        await sharp.default(buffer)
          .resize(sizeConfig.width)
          .webp({ quality: 90 })
          .toFile(webpPath);
        generated.push(webpPath);
      }
    }
    
    if (generated.length > 0) {
      console.log(`   📐 Responsive: ${generated.length} sizes generated`);
    }
    
  } catch (e) {
    // Responsive generation failed, continue
  }
  
  return generated;
}

// ═══════════════════════════════════════════════════════════════════════════
// PROFILE EXPORT (Generate all platform sizes from master)
// ═══════════════════════════════════════════════════════════════════════════

const PROFILE_SIZES = {
  'twitter': 800,
  'instagram': 640,
  'youtube': 800,
  'linkedin': 600,
  'tiktok': 400,
  'facebook': 640,
  'discord': 512,
  'favicon-512': 512,
  'favicon-256': 256,
  'favicon-192': 192,
  'favicon-128': 128,
  'favicon-64': 64,
  'favicon-32': 32,
  'apple-touch-icon': 180,
};

async function exportProfileSizes(masterPath: string): Promise<string[]> {
  const generated: string[] = [];
  
  try {
    const sharp = await import('sharp').catch(() => null);
    if (!sharp) {
      console.log('   ⚠️ Sharp not available, skipping profile exports');
      return [];
    }
    
    const buffer = fs.readFileSync(masterPath);
    const outputDir = path.join(process.cwd(), 'public/images/social');
    const brandDir = path.join(process.cwd(), 'public/images/brand');
    
    fs.mkdirSync(outputDir, { recursive: true });
    fs.mkdirSync(brandDir, { recursive: true });
    
    for (const [platform, size] of Object.entries(PROFILE_SIZES)) {
      const isFavicon = platform.startsWith('favicon') || platform === 'apple-touch-icon';
      const dir = isFavicon ? brandDir : outputDir;
      const filename = isFavicon ? `${platform}.png` : `profile-${platform}.png`;
      const outputPath = path.join(dir, filename);
      
      await sharp.default(buffer)
        .resize(size, size, { fit: 'cover' })
        .toFile(outputPath);
      
      generated.push(outputPath);
    }
    
    console.log(`   👤 Profile exports: ${generated.length} sizes`);
    
  } catch (e) {
    console.log(`   ⚠️ Profile export failed: ${e}`);
  }
  
  return generated;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('╔════════════════════════════════════════════════════════════════════╗');
  console.log('║  🎨 KELLY LORA ASSET FACTORY v2.0                                  ║');
  console.log('║  Enhanced with responsive sizes, WebP, and variants               ║');
  console.log('╚════════════════════════════════════════════════════════════════════╝');
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not found!');
    process.exit(1);
  }
  
  // Parse arguments
  const args = process.argv.slice(2);
  const priorityFilter = args.find(a => a.startsWith('--priority='))?.split('=')[1];
  const categoryFilter = args.find(a => a.startsWith('--category='))?.split('=')[1];
  const withVariants = args.includes('--with-variants');
  const withResponsive = args.includes('--responsive');
  const dryRun = args.includes('--dry-run');
  const skipExisting = args.includes('--skip-existing');
  
  // Filter assets
  let assetsToGenerate = ASSETS;
  
  if (priorityFilter) {
    assetsToGenerate = assetsToGenerate.filter(a => a.priority === priorityFilter);
    console.log(`\n🎯 Priority filter: ${priorityFilter}`);
  }
  
  if (categoryFilter) {
    assetsToGenerate = assetsToGenerate.filter(a => a.category === categoryFilter);
    console.log(`🎯 Category filter: ${categoryFilter}`);
  }
  
  // Check for existing files if skip-existing
  if (skipExisting) {
    const originalCount = assetsToGenerate.length;
    assetsToGenerate = assetsToGenerate.filter(a => {
      const fullPath = path.join(process.cwd(), a.output_path);
      return !fs.existsSync(fullPath);
    });
    console.log(`⏭️  Skipping ${originalCount - assetsToGenerate.length} existing files`);
  }
  
  // Calculate total generations including variants
  let totalGenerations = 0;
  for (const asset of assetsToGenerate) {
    if (withVariants && asset.variants) {
      totalGenerations += asset.variants;
    } else {
      totalGenerations += 1;
    }
  }
  
  console.log(`\n📊 Assets to generate: ${assetsToGenerate.length}`);
  console.log(`   Critical: ${assetsToGenerate.filter(a => a.priority === 'critical').length}`);
  console.log(`   High: ${assetsToGenerate.filter(a => a.priority === 'high').length}`);
  console.log(`   Medium: ${assetsToGenerate.filter(a => a.priority === 'medium').length}`);
  
  if (withVariants) {
    console.log(`\n🎲 With variants: ${totalGenerations} total generations`);
  }
  
  if (withResponsive) {
    console.log(`📐 Responsive sizes: Enabled`);
  }
  
  console.log(`\n🤖 Model: FLUX 1.1 Pro`);
  
  if (dryRun) {
    console.log('\n🔍 DRY RUN - Would generate:');
    for (const asset of assetsToGenerate) {
      const variantCount = withVariants && asset.variants ? asset.variants : 1;
      console.log(`   [${asset.priority}] ${asset.id} (${variantCount}x) → ${asset.output_path}`);
      if (asset.webp) console.log(`      + WebP version`);
      if (asset.responsive && withResponsive) console.log(`      + Responsive sizes`);
    }
    return;
  }
  
  // Generate assets
  const results: GenerationResult[] = [];
  const heygenAssets: GenerationResult[] = [];
  let profileMasterResult: GenerationResult | null = null;
  let completed = 0;
  
  for (const asset of assetsToGenerate) {
    const variantCount = withVariants && asset.variants ? asset.variants : 1;
    
    for (let v = 0; v < variantCount; v++) {
      const variant = variantCount > 1 ? v : undefined;
      const result = await generateWithFlux(asset, variant);
      results.push(result);
      completed++;
      
      console.log(`   📈 Progress: ${completed}/${totalGenerations}`);
      
      if (result.success) {
        // Track HeyGen uploads
        if (asset.heygen_upload) {
          heygenAssets.push(result);
        }
        
        // Track profile master for exports
        if (asset.id === 'profile-master' && variant === undefined) {
          profileMasterResult = result;
        }
        
        // Generate responsive sizes
        if (withResponsive && asset.responsive) {
          await generateResponsiveSizes(result);
        }
      }
      
      // Rate limit
      await new Promise(r => setTimeout(r, CONFIG.delayBetweenGenerations));
    }
  }
  
  // Export profile sizes from master
  if (profileMasterResult && profileMasterResult.localPath) {
    console.log('\n📤 Exporting profile sizes from master...');
    await exportProfileSizes(profileMasterResult.localPath);
  }
  
  // Summary
  console.log('\n\n' + '═'.repeat(70));
  console.log('📊 GENERATION SUMMARY');
  console.log('═'.repeat(70));
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`\n✅ Successful: ${successful.length}/${results.length}`);
  console.log(`❌ Failed: ${failed.length}/${results.length}`);
  
  if (failed.length > 0) {
    console.log('\n🔴 Failed assets:');
    for (const f of failed) {
      console.log(`   - ${f.asset.id}: ${f.error}`);
    }
  }
  
  if (heygenAssets.length > 0) {
    console.log(`\n🎬 HeyGen-ready assets: ${heygenAssets.length}`);
    console.log('   Run heygen-upload-avatars.ts to upload these as talking photos');
  }
  
  // Count additional files
  let additionalFiles = 0;
  for (const r of successful) {
    additionalFiles += r.additionalPaths?.length || 0;
  }
  if (additionalFiles > 0) {
    console.log(`📦 Additional formats: ${additionalFiles} (WebP, responsive)`);
  }
  
  // Save manifest
  const manifest = {
    generated: new Date().toISOString(),
    model: 'black-forest-labs/flux-1.1-pro',
    total: results.length,
    successful: successful.length,
    failed: failed.length,
    withVariants,
    withResponsive,
    assets: results.map(r => ({
      id: r.asset.id,
      variant: r.variant,
      success: r.success,
      path: r.localPath,
      additionalPaths: r.additionalPaths,
      heygen_upload: r.asset.heygen_upload,
      duration: r.duration,
      error: r.error,
    })),
  };
  
  const manifestPath = path.join(process.cwd(), 'generated-assets-manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\n💾 Manifest saved: ${manifestPath}`);
  
  // Estimate cost
  const costPerImage = 0.04; // FLUX 1.1 Pro approximate cost
  const estimatedCost = results.length * costPerImage;
  console.log(`💰 Estimated cost: $${estimatedCost.toFixed(2)}`);
  
  console.log('\n' + '═'.repeat(70));
}

main().catch(console.error);
