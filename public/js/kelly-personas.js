/**
 * Kelly Teaching Personas - Unified Asset Module
 * Production-ready persona definitions with CDN-hosted images
 * 
 * Usage:
 *   import { PERSONAS, getPersonaImage, SUPABASE_CDN } from './kelly-personas.js';
 *   
 *   // Get image URL for a persona
 *   const scientistUrl = getPersonaImage('scientist', 'head');
 *   
 *   // Iterate all personas
 *   PERSONAS.forEach(p => console.log(p.name, p.icon));
 */

// Supabase CDN base URL for Kelly assets
export const SUPABASE_CDN = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates';

// All 12 Teaching Personas with full metadata
export const PERSONAS = [
    { 
        id: 'scientist',
        name: 'The Scientist', 
        icon: '🔬', 
        tagline: 'Data-driven precision',
        description: 'Lab goggles on forehead',
        color: '#3b82f6',
        images: {
            head: 'heygen/archetypes-head-only/kelly_scientist_head.png',
            clean: 'heygen/archetypes-head-only/kelly_scientist_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_scientist_prop_head.png'
        }
    },
    { 
        id: 'explorer',
        name: 'The Explorer', 
        icon: '🧭', 
        tagline: 'Wonder and discovery',
        description: 'Aviator goggles + bandana',
        color: '#eab308',
        images: {
            head: 'heygen/archetypes-head-only/kelly_explorer_head.png',
            clean: 'heygen/archetypes-head-only/kelly_explorer_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_explorer_prop_head.png'
        }
    },
    { 
        id: 'rebel',
        name: 'The Rebel', 
        icon: '⚡', 
        tagline: 'Bold challenging spirit',
        description: 'Sunglasses in hair + earring',
        color: '#ef4444',
        images: {
            head: 'heygen/archetypes-head-only/kelly_rebel_head.png',
            clean: 'heygen/archetypes-head-only/kelly_rebel_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_rebel_prop_head.png'
        }
    },
    { 
        id: 'architect',
        name: 'The Architect', 
        icon: '🏛️', 
        tagline: 'Methodical structure',
        description: 'Pencil behind ear + glasses',
        color: '#6b7280',
        images: {
            head: 'heygen/archetypes-head-only/kelly_architect_head.png',
            clean: 'heygen/archetypes-head-only/kelly_architect_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_architect_prop_head.png'
        }
    },
    { 
        id: 'diplomat',
        name: 'The Diplomat', 
        icon: '🤝', 
        tagline: 'Inclusive harmony',
        description: 'Pearl studs + velvet headband',
        color: '#22c55e',
        images: {
            head: 'heygen/archetypes-head-only/kelly_diplomat_head.png',
            clean: 'heygen/archetypes-head-only/kelly_diplomat_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_diplomat_prop_head.png'
        }
    },
    { 
        id: 'empath',
        name: 'The Empath', 
        icon: '💗', 
        tagline: 'Nurturing warmth',
        description: 'Pink headband + lavender',
        color: '#ec4899',
        images: {
            head: 'heygen/archetypes-head-only/kelly_empath_head.png',
            clean: 'heygen/archetypes-head-only/kelly_empath_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_empath_prop_head.png'
        }
    },
    { 
        id: 'macgyver',
        name: 'The MacGyver', 
        icon: '🔧', 
        tagline: 'Hands-on problem solver',
        description: 'Shop glasses + red bandana',
        color: '#f97316',
        images: {
            head: 'heygen/archetypes-head-only/kelly_macgyver_head.png',
            clean: 'heygen/archetypes-head-only/kelly_macgyver_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_macgyver_prop_head.png'
        }
    },
    { 
        id: 'mystic',
        name: 'The Mystic', 
        icon: '✨', 
        tagline: 'Profound serenity',
        description: 'Third eye amethyst + gold chain',
        color: '#a855f7',
        images: {
            head: 'heygen/archetypes-head-only/kelly_mystic_head.png',
            clean: 'heygen/archetypes-head-only/kelly_mystic_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_mystic_prop_head.png'
        }
    },
    { 
        id: 'provider',
        name: 'The Provider', 
        icon: '🛡️', 
        tagline: 'Reassuring strength',
        description: 'Cream knit headband',
        color: '#14b8a6',
        images: {
            head: 'heygen/archetypes-head-only/kelly_provider_head.png',
            clean: 'heygen/archetypes-head-only/kelly_provider_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_provider_prop_head.png'
        }
    },
    { 
        id: 'storyteller',
        name: 'The Storyteller', 
        icon: '📖', 
        tagline: 'Theatrical captivation',
        description: 'Gold glasses + peacock feather',
        color: '#f472b6',
        images: {
            head: 'heygen/archetypes-head-only/kelly_storyteller_head.png',
            clean: 'heygen/archetypes-head-only/kelly_storyteller_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_storyteller_prop_head.png'
        }
    },
    { 
        id: 'strategist',
        name: 'The Strategist', 
        icon: '🎯', 
        tagline: 'Sharp tactical mind',
        description: 'Angular glasses + chess clip',
        color: '#6366f1',
        images: {
            head: 'heygen/archetypes-head-only/kelly_strategist_head.png',
            clean: 'heygen/archetypes-head-only/kelly_strategist_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_strategist_prop_head.png'
        }
    },
    { 
        id: 'survivor',
        name: 'The Survivor', 
        icon: '🏕️', 
        tagline: 'Grounded resilience',
        description: 'Military bandana + dog tags',
        color: '#84cc16',
        images: {
            head: 'heygen/archetypes-head-only/kelly_survivor_head.png',
            clean: 'heygen/archetypes-head-only/kelly_survivor_clean.png',
            prop: 'heygen/archetypes-head-only/kelly_survivor_prop_head.png'
        }
    }
];

// Map for quick lookup by ID
export const PERSONAS_MAP = Object.fromEntries(PERSONAS.map(p => [p.id, p]));

// Ordered list of persona IDs
export const PERSONA_ORDER = PERSONAS.map(p => p.id);

// Age variants for persona "head" images (60 base heads = 12 personas × 5 ages)
export const AGE_VARIANTS = ['kid', 'teen', 'adult', 'elder', 'super_elder'];

/**
 * Get the full CDN URL for a persona image
 * @param {string} personaId - e.g., 'scientist', 'explorer'
 * @param {string} variant - 'head' | 'clean' | 'prop' (default: 'head')
 * @param {string} ageVariant - 'kid' | 'teen' | 'adult' | 'elder' | 'super_elder' (default: 'adult')
 * @returns {string} Full CDN URL
 */
export function getPersonaImage(personaId, variant = 'head', ageVariant = 'adult') {
    const persona = PERSONAS_MAP[personaId];
    if (!persona) {
        console.warn(`Unknown persona: ${personaId}`);
        return `${SUPABASE_CDN}/heygen/archetypes-head-only/kelly_scientist_head.png`;
    }

    // Age variants currently exist only for the head-only archetype images.
    // For clean/prop (and unknown ages), we intentionally fall back to the adult/base set.
    if (variant === 'head' && ageVariant && ageVariant !== 'adult') {
        if (!AGE_VARIANTS.includes(ageVariant)) {
            console.warn(`Unknown ageVariant: ${ageVariant} (falling back to adult)`);
        } else {
            return `${SUPABASE_CDN}/heygen/archetypes-head-only/age/${ageVariant}/kelly_${personaId}_head.png`;
        }
    }

    const imagePath = persona.images[variant] || persona.images.head;
    return `${SUPABASE_CDN}/${imagePath}`;
}

/**
 * Get persona by ID
 * @param {string} personaId 
 * @returns {Object|null}
 */
export function getPersona(personaId) {
    return PERSONAS_MAP[personaId] || null;
}

/**
 * Get a random persona
 * @returns {Object}
 */
export function getRandomPersona() {
    return PERSONAS[Math.floor(Math.random() * PERSONAS.length)];
}

/**
 * Preload all persona images for smooth transitions
 * @param {string} variant - 'head' | 'clean' | 'prop'
 * @returns {Promise<void>}
 */
export async function preloadPersonaImages(variant = 'head') {
    const promises = PERSONAS.map(persona => {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = resolve;
            img.onerror = reject;
            img.src = getPersonaImage(persona.id, variant);
        });
    });
    await Promise.allSettled(promises);
}

// For non-module usage (script tag)
if (typeof window !== 'undefined') {
    window.KellyPersonas = {
        PERSONAS,
        PERSONAS_MAP,
        PERSONA_ORDER,
        AGE_VARIANTS,
        SUPABASE_CDN,
        getPersonaImage,
        getPersona,
        getRandomPersona,
        preloadPersonaImages
    };
}
