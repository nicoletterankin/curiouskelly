/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * GOLDEN V2 - LESSON DNA GENERATOR (MULTILINGUAL EDITION)
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Generates complete lesson DNA for all 365 days across all 6 age buckets.
 * Includes NATIVE templates for English, Spanish, and French.
 * 
 * @version 2.1.0 - Polyglot Update
 * @author Curious Kelly AI
 */

import fs from 'fs';
import path from 'path';

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const CONFIG = {
  outputDir: './generated/lessons',
  lessonsPerBatch: 10,
  totalLessons: 365,
  
  ageBuckets: [
    { id: '2-5', name: 'Little Learner', persona: 'Playful Friend', style: 'Story-based, fun, simple words' },
    { id: '6-12', name: 'Curious Explorer', persona: 'Cool Big Sister', style: 'Hands-on, engaging, curious' },
    { id: '13-17', name: 'Teen Scholar', persona: 'Smart Mentor', style: 'Direct, relatable, no fluff' },
    { id: '18-35', name: 'Adult Learner', persona: 'Equal Partner', style: 'Practical, clear, conversational' },
    { id: '36-60', name: 'Seasoned Mind', persona: 'Respectful Guide', style: 'Efficient, substantive' },
    { id: '61-102', name: 'Wisdom Keeper', persona: 'Warm Companion', style: 'Warm, thoughtful, reflective' }
  ],
  
  phases: ['hook', 'q1', 'q2', 'q3', 'wisdom']
};

// ═══════════════════════════════════════════════════════════════════════════════
// UNIVERSAL TOPICS (With Translations)
// ═══════════════════════════════════════════════════════════════════════════════

const UNIVERSAL_TOPICS = [
  // ═══════════════════════════════════════════════════════════════════════════════
  // DAYS 1-10: NATURE & BODY BASICS
  // ═══════════════════════════════════════════════════════════════════════════════
  { 
    day: 1, 
    topic: { en: 'The Sun', es: 'El Sol', fr: 'Le Soleil' },
    truth: { en: 'Our star gives life to everything on Earth', es: 'Nuestra estrella da vida a todo en la Tierra', fr: 'Notre étoile donne la vie à tout sur Terre' }
  },
  { 
    day: 2, 
    topic: { en: 'Why the Sky is Blue', es: 'Por qué el cielo es azul', fr: 'Pourquoi le ciel est bleu' },
    truth: { en: 'Light scatters through our atmosphere in beautiful ways', es: 'La luz se dispersa por nuestra atmósfera de formas hermosas', fr: 'La lumière se disperse dans notre atmosphère de manière magnifique' }
  },
  { 
    day: 3, 
    topic: { en: 'How Seeds Grow', es: 'Cómo crecen las semillas', fr: 'Comment poussent les graines' },
    truth: { en: 'Life begins small and grows with patience', es: 'La vida comienza pequeña y crece con paciencia', fr: 'La vie commence petite et grandit avec patience' }
  },
  { 
    day: 4, 
    topic: { en: 'The Water Cycle', es: 'El ciclo del agua', fr: "Le cycle de l'eau" },
    truth: { en: 'Water travels endlessly through our world', es: 'El agua viaja sin fin a través de nuestro mundo', fr: "L'eau voyage sans fin à travers notre monde" }
  },
  { 
    day: 5, 
    topic: { en: 'Why We Sleep', es: 'Por qué dormimos', fr: 'Pourquoi nous dormons' },
    truth: { en: 'Rest rebuilds our bodies and minds', es: 'El descanso reconstruye nuestros cuerpos y mentes', fr: 'Le repos reconstruit nos corps et nos esprits' }
  },
  { 
    day: 6, 
    topic: { en: 'How Birds Fly', es: 'Cómo vuelan las aves', fr: 'Comment volent les oiseaux' },
    truth: { en: 'Nature engineered flight millions of years ago', es: 'La naturaleza diseñó el vuelo hace millones de años', fr: 'La nature a conçu le vol il y a des millions d\'années' }
  },
  { 
    day: 7, 
    topic: { en: 'The Moon', es: 'La Luna', fr: 'La Lune' },
    truth: { en: 'Our constant companion lights the night', es: 'Nuestra compañera constante ilumina la noche', fr: 'Notre compagnon constant illumine la nuit' }
  },
  { 
    day: 8, 
    topic: { en: 'Why Leaves Change Color', es: 'Por qué las hojas cambian de color', fr: 'Pourquoi les feuilles changent de couleur' },
    truth: { en: 'Trees prepare for winter in brilliant display', es: 'Los árboles se preparan para el invierno con una exhibición brillante', fr: 'Les arbres se préparent pour l\'hiver dans un affichage brillant' }
  },
  { 
    day: 9, 
    topic: { en: 'How Sound Works', es: 'Cómo funciona el sonido', fr: 'Comment fonctionne le son' },
    truth: { en: 'Vibrations carry meaning through the air', es: 'Las vibraciones transportan significado a través del aire', fr: 'Les vibrations transportent du sens à travers l\'air' }
  },
  { 
    day: 10, 
    topic: { en: 'The Heart', es: 'El Corazón', fr: 'Le Cœur' },
    truth: { en: 'This tireless pump never stops working for us', es: 'Esta bomba incansable nunca deja de trabajar para nosotros', fr: 'Cette pompe infatigable ne cesse jamais de travailler pour nous' }
  },
  // ═══════════════════════════════════════════════════════════════════════════════
  // DAYS 11-20: BODY, TECHNOLOGY & EARTH
  // ═══════════════════════════════════════════════════════════════════════════════
  { 
    day: 11, 
    topic: { en: 'Why We Yawn', es: 'Por qué bostezamos', fr: 'Pourquoi nous bâillons' },
    truth: { en: 'Our body has automatic systems we barely notice', es: 'Nuestro cuerpo tiene sistemas automáticos que apenas notamos', fr: 'Notre corps a des systèmes automatiques que nous remarquons à peine' }
  },
  { 
    day: 12, 
    topic: { en: 'How Rainbows Form', es: 'Cómo se forman los arcoíris', fr: 'Comment se forment les arcs-en-ciel' },
    truth: { en: 'Light reveals hidden colors when it bends', es: 'La luz revela colores ocultos cuando se dobla', fr: 'La lumière révèle des couleurs cachées quand elle se courbe' }
  },
  { 
    day: 13, 
    topic: { en: 'The Brain', es: 'El Cerebro', fr: 'Le Cerveau' },
    truth: { en: 'The most complex object in the universe sits between your ears', es: 'El objeto más complejo del universo está entre tus oídos', fr: "L'objet le plus complexe de l'univers se trouve entre vos oreilles" }
  },
  { 
    day: 14, 
    topic: { en: 'Why Ice Floats', es: 'Por qué flota el hielo', fr: 'Pourquoi la glace flotte' },
    truth: { en: 'Water breaks the rules of physics to protect life', es: 'El agua rompe las reglas de la física para proteger la vida', fr: 'L\'eau enfreint les règles de la physique pour protéger la vie' }
  },
  { 
    day: 15, 
    topic: { en: 'How Computers Think', es: 'Cómo piensan las computadoras', fr: 'Comment pensent les ordinateurs' },
    truth: { en: 'Billions of tiny switches make decisions', es: 'Miles de millones de pequeños interruptores toman decisiones', fr: 'Des milliards de petits interrupteurs prennent des décisions' }
  },
  { 
    day: 16, 
    topic: { en: 'Gravity', es: 'La Gravedad', fr: 'La Gravité' },
    truth: { en: 'An invisible force shapes everything from galaxies to raindrops', es: 'Una fuerza invisible da forma a todo, desde galaxias hasta gotas de lluvia', fr: 'Une force invisible façonne tout, des galaxies aux gouttes de pluie' }
  },
  { 
    day: 17, 
    topic: { en: 'Why We Dream', es: 'Por qué soñamos', fr: 'Pourquoi nous rêvons' },
    truth: { en: 'Our sleeping mind processes and creates worlds', es: 'Nuestra mente dormida procesa y crea mundos', fr: 'Notre esprit endormi traite et crée des mondes' }
  },
  { 
    day: 18, 
    topic: { en: 'The Ocean', es: 'El Océano', fr: "L'Océan" },
    truth: { en: 'Most of our planet remains unexplored beneath the waves', es: 'La mayor parte de nuestro planeta permanece inexplorada bajo las olas', fr: 'La majeure partie de notre planète reste inexplorée sous les vagues' }
  },
  { 
    day: 19, 
    topic: { en: 'How Plants Make Food', es: 'Cómo las plantas fabrican alimento', fr: 'Comment les plantes fabriquent leur nourriture' },
    truth: { en: 'Sunlight becomes sugar through an ancient miracle', es: 'La luz del sol se convierte en azúcar a través de un antiguo milagro', fr: 'La lumière du soleil devient du sucre par un miracle ancien' }
  },
  { 
    day: 20, 
    topic: { en: 'Electricity', es: 'La Electricidad', fr: "L'Électricité" },
    truth: { en: 'Tiny particles in motion power our modern world', es: 'Pequeñas partículas en movimiento alimentan nuestro mundo moderno', fr: 'De minuscules particules en mouvement alimentent notre monde moderne' }
  },
  // ═══════════════════════════════════════════════════════════════════════════════
  // DAYS 21-30: SPACE, ANIMALS & COMMUNICATION
  // ═══════════════════════════════════════════════════════════════════════════════
  { 
    day: 21, 
    topic: { en: 'The Stars', es: 'Las Estrellas', fr: 'Les Étoiles' },
    truth: { en: 'Every atom in your body was forged inside a star', es: 'Cada átomo de tu cuerpo fue forjado dentro de una estrella', fr: 'Chaque atome de votre corps a été forgé à l\'intérieur d\'une étoile' }
  },
  { 
    day: 22, 
    topic: { en: 'How Fish Breathe Underwater', es: 'Cómo respiran los peces bajo el agua', fr: "Comment les poissons respirent sous l'eau" },
    truth: { en: 'Different solutions exist for the same challenge', es: 'Existen diferentes soluciones para el mismo desafío', fr: 'Différentes solutions existent pour le même défi' }
  },
  { 
    day: 23, 
    topic: { en: 'The Internet', es: 'Internet', fr: 'Internet' },
    truth: { en: 'Information travels at the speed of light connecting everyone', es: 'La información viaja a la velocidad de la luz conectando a todos', fr: "L'information voyage à la vitesse de la lumière en connectant tout le monde" }
  },
  { 
    day: 24, 
    topic: { en: 'Why We Laugh', es: 'Por qué nos reímos', fr: 'Pourquoi nous rions' },
    truth: { en: 'Joy is a universal language we all understand', es: 'La alegría es un lenguaje universal que todos entendemos', fr: 'La joie est un langage universel que nous comprenons tous' }
  },
  { 
    day: 25, 
    topic: { en: 'Dinosaurs', es: 'Los Dinosaurios', fr: 'Les Dinosaures' },
    truth: { en: 'Giants once walked where we walk today', es: 'Los gigantes una vez caminaron donde caminamos hoy', fr: 'Des géants ont autrefois marché là où nous marchons aujourd\'hui' }
  },
  { 
    day: 26, 
    topic: { en: 'How Airplanes Fly', es: 'Cómo vuelan los aviones', fr: 'Comment volent les avions' },
    truth: { en: 'Shape and speed create lift that defies gravity', es: 'La forma y la velocidad crean sustentación que desafía la gravedad', fr: 'La forme et la vitesse créent une portance qui défie la gravité' }
  },
  { 
    day: 27, 
    topic: { en: 'The Amazon Rainforest', es: 'La Selva Amazónica', fr: 'La Forêt Amazonienne' },
    truth: { en: 'Half of all species live in one place on Earth', es: 'La mitad de todas las especies viven en un solo lugar de la Tierra', fr: 'La moitié de toutes les espèces vivent au même endroit sur Terre' }
  },
  { 
    day: 28, 
    topic: { en: 'How We See Colors', es: 'Cómo vemos los colores', fr: 'Comment nous voyons les couleurs' },
    truth: { en: 'Our eyes and brain work together to paint the world', es: 'Nuestros ojos y cerebro trabajan juntos para pintar el mundo', fr: 'Nos yeux et notre cerveau travaillent ensemble pour peindre le monde' }
  },
  { 
    day: 29, 
    topic: { en: 'Volcanoes', es: 'Los Volcanes', fr: 'Les Volcans' },
    truth: { en: 'The Earth is alive and constantly reshaping itself', es: 'La Tierra está viva y constantemente se transforma', fr: 'La Terre est vivante et se remodèle constamment' }
  },
  { 
    day: 30, 
    topic: { en: 'Why We Age', es: 'Por qué envejecemos', fr: 'Pourquoi nous vieillissons' },
    truth: { en: 'Time writes its story on our cells', es: 'El tiempo escribe su historia en nuestras células', fr: 'Le temps écrit son histoire sur nos cellules' }
  }
];

// Fill remaining days with Generated Content
// Note: In production, these would be pulled from a larger localized database
for (let day = UNIVERSAL_TOPICS.length + 1; day <= 365; day++) {
  const topicCategories = [
    { 
      topic: { en: 'How Computers Think', es: 'Cómo piensan las computadoras', fr: 'Comment pensent les ordinateurs' }, 
      truth: { en: 'Billions of tiny switches make decisions', es: 'Miles de millones de pequeños interruptores toman decisiones', fr: 'Des milliards de petits interrupteurs prennent des décisions' }
    },
    { 
      topic: { en: 'The Internet', es: 'Internet', fr: 'Internet' }, 
      truth: { en: 'Information travels at the speed of light', es: 'La información viaja a la velocidad de la luz', fr: "L'information voyage à la vitesse de la lumière" }
    },
    {
      topic: { en: 'Why We Age', es: 'Por qué envejecemos', fr: 'Pourquoi nous vieillissons' },
      truth: { en: 'Time writes its story on our cells', es: 'El tiempo escribe su historia en nuestras células', fr: 'Le temps écrit son histoire sur nos cellules' }
    },
    {
      topic: { en: 'How Airplanes Fly', es: 'Cómo vuelan los aviones', fr: 'Comment volent les avions' },
      truth: { en: 'Shape and speed create lift', es: 'La forma y la velocidad crean sustentación', fr: 'La forme et la vitesse créent la portance' }
    },
    {
      topic: { en: 'The Amazon Rainforest', es: 'La selva amazónica', fr: 'La forêt amazonienne' },
      truth: { en: 'Half of all species live in one place', es: 'La mitad de todas las especies viven en un solo lugar', fr: 'La moitié de toutes les espèces vivent au même endroit' }
    }
  ];
  
  const category = topicCategories[(day - 1) % topicCategories.length];
  UNIVERSAL_TOPICS.push({
    day,
    topic: {
        en: `${category.topic.en} (Day ${day})`,
        es: `${category.topic.es} (Día ${day})`,
        fr: `${category.topic.fr} (Jour ${day})`
    },
    truth: category.truth
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTILINGUAL TEMPLATES
// ═══════════════════════════════════════════════════════════════════════════════

const TEMPLATES = {
  // ENGLISH (Source)
  en: {
    '2-5': {
      hook: (topic) => `Hi little friend! 🌟 Today we're going to learn about something super cool - ${topic}! Are you ready for an adventure?`,
      q1: (topic, fact) => `Did you know? ${fact} Isn't that amazing? Let's learn more!`,
      q2: (topic, fact) => `Here's something fun! ${fact} Can you imagine that?`,
      q3: (topic, fact) => `Wow! ${fact} You're getting so smart!`,
      wisdom: (topic, truth) => `Remember, little learner: ${truth} You learned something wonderful today! ⭐`
    },
    '18-35': {
      hook: (topic) => `Today we're covering ${topic}. Practical knowledge you can actually use.`,
      q1: (topic, fact) => `Let's start with the basics. ${fact}`,
      q2: (topic, fact) => `Building on that: ${fact}`,
      q3: (topic, fact) => `And here's the practical application. ${fact}`,
      wisdom: (topic, truth) => `The takeaway: ${truth} Knowledge that makes a difference.`
    }
  },

  // SPANISH
  es: {
    '2-5': {
      hook: (topic) => `¡Hola pequeño amigo! 🌟 Hoy vamos a aprender algo súper genial: ¡${topic}! ¿Estás listo para una aventura?`,
      q1: (topic, fact) => `¿Sabías que? ${fact} ¿No es asombroso? ¡Aprendamos más!`,
      q2: (topic, fact) => `¡Aquí hay algo divertido! ${fact} ¿Te lo puedes imaginar?`,
      q3: (topic, fact) => `¡Guau! ${fact} ¡Te estás volviendo muy inteligente!`,
      wisdom: (topic, truth) => `Recuerda, pequeño aprendiz: ${truth} ¡Hoy aprendiste algo maravilloso! ⭐`
    },
    '18-35': {
      hook: (topic) => `Hoy cubriremos ${topic}. Conocimiento práctico que realmente puedes usar.`,
      q1: (topic, fact) => `Comencemos con lo básico. ${fact}`,
      q2: (topic, fact) => `Profundizando en eso: ${fact}`,
      q3: (topic, fact) => `Y aquí está la aplicación práctica. ${fact}`,
      wisdom: (topic, truth) => `La conclusión: ${truth} Conocimiento que marca la diferencia.`
    }
  },

  // FRENCH
  fr: {
    '2-5': {
      hook: (topic) => `Salut petit ami ! 🌟 Aujourd'hui, nous allons apprendre quelque chose de super cool : ${topic} ! Es-tu prêt pour l'aventure ?`,
      q1: (topic, fact) => `Savais-tu que ? ${fact} N'est-ce pas incroyable ? Apprenons-en plus !`,
      q2: (topic, fact) => `Voici quelque chose d'amusant ! ${fact} Peux-tu imaginer cela ?`,
      q3: (topic, fact) => `Waouh ! ${fact} Tu deviens si intelligent !`,
      wisdom: (topic, truth) => `Souviens-toi, petit apprenant : ${truth} Tu as appris quelque chose de merveilleux aujourd'hui ! ⭐`
    },
    '18-35': {
      hook: (topic) => `Aujourd'hui, nous abordons ${topic}. Des connaissances pratiques que vous pouvez vraiment utiliser.`,
      q1: (topic, fact) => `Commençons par les bases. ${fact}`,
      q2: (topic, fact) => `En s'appuyant là-dessus : ${fact}`,
      q3: (topic, fact) => `Et voici l'application pratique. ${fact}`,
      wisdom: (topic, truth) => `Ce qu'il faut retenir : ${truth} Le savoir qui fait la différence.`
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// FACT GENERATION (Multilingual)
// ═══════════════════════════════════════════════════════════════════════════════

function generateFacts(topicObj, truthObj, lang) {
  const topic = topicObj[lang];
  const truth = truthObj[lang];

  // Hardcoded facts for demo topics (Day 1)
  if (topicObj.en === 'The Sun') {
    if (lang === 'es') return [
      'El Sol está a 93 millones de millas de distancia, pero su luz llega a la Tierra en solo 8 minutos',
      'El Sol es tan grande que un millón de Tierras podrían caber dentro de él',
      'Cada segundo, el Sol convierte 4 millones de toneladas de materia en energía pura'
    ];
    if (lang === 'fr') return [
      'Le Soleil est à 150 millions de kilomètres, pourtant sa lumière atteint la Terre en seulement 8 minutes',
      'Le Soleil est si grand qu\'un million de Terres pourraient tenir à l\'intérieur',
      'Chaque seconde, le Soleil convertit 4 millions de tonnes de matière en énergie pure'
    ];
    // Default English
    return [
      'The Sun is 93 million miles away, yet its light reaches Earth in just 8 minutes',
      'The Sun is so big that one million Earths could fit inside it',
      'Every second, the Sun converts 4 million tons of matter into pure energy'
    ];
  }

  // Generic generator for other topics
  if (lang === 'es') {
    return [
      `Lo primero que hay que entender sobre ${topic.toLowerCase()} es que afecta nuestra vida diaria de formas que a menudo no notamos`,
      `Los científicos han descubierto patrones fascinantes en cómo funciona ${topic.toLowerCase()} en la naturaleza`,
      `${truth} - y este principio se conecta con muchas otras áreas del conocimiento`
    ];
  }
  if (lang === 'fr') {
    return [
      `La première chose à comprendre à propos de ${topic.toLowerCase()} est qu'il affecte notre vie quotidienne de manières que nous ne remarquons souvent pas`,
      `Les scientifiques ont découvert des modèles fascinants dans la façon dont ${topic.toLowerCase()} fonctionne dans la nature`,
      `${truth} - et ce principe se connecte à tant d'autres domaines de la connaissance`
    ];
  }

  return [
    `The first thing to understand about ${topic.toLowerCase()} is that it affects our daily lives in ways we often don't notice`,
    `Scientists have discovered fascinating patterns in how ${topic.toLowerCase()} works in nature`,
    `${truth} - and this principle connects to so many other areas of knowledge`
  ];
}

// ═══════════════════════════════════════════════════════════════════════════════
// LESSON GENERATOR CLASS
// ═══════════════════════════════════════════════════════════════════════════════

class LessonDNAGenerator {
  constructor(config = {}) {
    this.config = { ...CONFIG, ...config };
  }
  
  generateLessonDNA(dayNumber) {
    const topicData = UNIVERSAL_TOPICS[dayNumber - 1];
    if (!topicData) throw new Error(`No topic found for day ${dayNumber}`);
    
    // Base metadata (using English for system fields)
    const lessonDNA = {
      meta: {
        day: dayNumber,
        topic: topicData.topic.en,
        universalTruth: topicData.truth.en,
        generatedAt: new Date().toISOString(),
        version: '2.1.0-polyglot'
      },
      visuals: {
        // Visuals are language-agnostic descriptions for the generator
        hook: `Kelly presenting ${topicData.topic.en} with an engaging, welcoming expression`,
        q1: `Kelly explaining the first concept about ${topicData.topic.en} with curious gesture`,
        q2: `Kelly diving deeper into ${topicData.topic.en} with animated expression`,
        q3: `Kelly revealing the fascinating aspect of ${topicData.topic.en}`,
        wisdom: `Kelly reflecting warmly on the life lesson from ${topicData.topic.en}`
      },
      ageVariants: {}
    };
    
    // Generate content for each age bucket
    for (const bucket of this.config.ageBuckets) {
      // Get templates for this age, fallback to '18-35' if specific age not defined in template
      const getTemplate = (lang) => TEMPLATES[lang][bucket.id] || TEMPLATES[lang]['18-35'];

      const contentEn = this.generateContentForLang('en', bucket, topicData, getTemplate('en'));
      const contentEs = this.generateContentForLang('es', bucket, topicData, getTemplate('es'));
      const contentFr = this.generateContentForLang('fr', bucket, topicData, getTemplate('fr'));

      lessonDNA.ageVariants[bucket.id] = {
        persona: bucket.persona,
        teachingStyle: bucket.style,
        phases: {
          hook: { en: contentEn.hook, es: contentEs.hook, fr: contentFr.hook },
          q1: { en: contentEn.q1, es: contentEs.q1, fr: contentFr.q1 },
          q2: { en: contentEn.q2, es: contentEs.q2, fr: contentFr.q2 },
          q3: { en: contentEn.q3, es: contentEs.q3, fr: contentFr.q3 },
          wisdom: { en: contentEn.wisdom, es: contentEs.wisdom, fr: contentFr.wisdom }
        },
        durations: {
          hook: this.estimateDuration(contentEn.hook),
          q1: this.estimateDuration(contentEn.q1),
          q2: this.estimateDuration(contentEn.q2),
          q3: this.estimateDuration(contentEn.q3),
          wisdom: this.estimateDuration(contentEn.wisdom)
        },
        voiceSettings: this.getVoiceSettings(bucket.id)
      };
    }
    
    return lessonDNA;
  }

  generateContentForLang(lang, bucket, topicData, template) {
    const facts = generateFacts(topicData.topic, topicData.truth, lang);
    const topic = topicData.topic[lang];
    const truth = topicData.truth[lang];

    return {
      hook: template.hook(topic),
      q1: template.q1(topic, facts[0]),
      q2: template.q2(topic, facts[1]),
      q3: template.q3(topic, facts[2]),
      wisdom: template.wisdom(topic, truth)
    };
  }

  estimateDuration(text) {
    const words = text.split(/\s+/).length;
    return Math.ceil((words / 150) * 60);
  }
  
  getVoiceSettings(bucketId) {
    const settings = {
      '2-5':   { pitch: 1.15, speed: 0.95, warmth: 'high', energy: 'playful' },
      '6-12':  { pitch: 1.08, speed: 1.0, warmth: 'medium-high', energy: 'enthusiastic' },
      '13-17': { pitch: 1.0, speed: 1.05, warmth: 'medium', energy: 'direct' },
      '18-35': { pitch: 1.0, speed: 1.0, warmth: 'medium', energy: 'conversational' },
      '36-60': { pitch: 0.95, speed: 0.95, warmth: 'medium', energy: 'measured' },
      '61-102': { pitch: 0.92, speed: 0.9, warmth: 'high', energy: 'gentle' }
    };
    return settings[bucketId] || settings['18-35'];
  }
  
  async generateAllLessons() {
    const outputDir = this.config.outputDir;
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
    
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  GOLDEN V2.1 - POLYGLOT DNA GENERATOR');
    console.log('  Generating 365 lessons with EN, ES, FR content');
    console.log('═══════════════════════════════════════════════════════════════');
    
    const lessons = [];
    for (let day = 1; day <= this.config.totalLessons; day++) {
      const lessonDNA = this.generateLessonDNA(day);
      lessons.push(lessonDNA);
      const paddedDay = String(day).padStart(3, '0');
      fs.writeFileSync(path.join(outputDir, `day-${paddedDay}.json`), JSON.stringify(lessonDNA, null, 2));
      if (day % 30 === 0) console.log(`  ✓ Generated days 1-${day} (${Math.round(day/365*100)}%)`);
    }
    
    fs.writeFileSync(path.join(outputDir, 'manifest.json'), JSON.stringify({
      version: '2.1.0-polyglot',
      generatedAt: new Date().toISOString(),
      totalLessons: lessons.length,
      ageBuckets: this.config.ageBuckets.map(b => b.id),
      phases: this.config.phases,
      lessons: lessons.map(l => ({ day: l.meta.day, topic: l.meta.topic }))
    }, null, 2));
    
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`  ✅ COMPLETE: ${lessons.length} lessons generated`);
    console.log('═══════════════════════════════════════════════════════════════');
    return lessons;
  }
}

export { LessonDNAGenerator, CONFIG, UNIVERSAL_TOPICS, TEMPLATES };
export default LessonDNAGenerator;

if (process.argv[1].includes('lesson-dna-generator')) {
  new LessonDNAGenerator().generateAllLessons().catch(console.error);
}
