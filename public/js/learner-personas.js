/**
 * Learner Personas - Rich, Diverse Global Learning Community
 * ============================================================
 * 
 * PHILOSOPHY:
 * These personas represent the diversity of human learning. Each has:
 * - A real name from their culture
 * - An age and life stage
 * - A learning style/personality
 * - An avatar image
 * 
 * TRUST & SAFETY:
 * All personas are clearly simulated. ✨ marks all content.
 * Purpose: Create belonging, normalize struggle, show learning is universal.
 * 
 * IMAGE SYSTEM:
 * Avatar images are diverse, AI-generated portraits that represent
 * global learners. Stored in /images/learners/[id].jpg
 */

// ═══════════════════════════════════════════════════════════════════
// LEARNER PERSONAS - 60 diverse, authentic characters
// ═══════════════════════════════════════════════════════════════════

const LEARNER_PERSONAS = [
  // === NORTH AMERICA ===
  { id: "emma-us", name: "Emma", age: 28, country: "US", flag: "🇺🇸", 
    avatar: "/images/learners/emma-us.jpg",
    bio: "Software engineer, learns during lunch breaks",
    style: "analytical", ageGroup: "adult" },
  
  { id: "marcus-us", name: "Marcus", age: 16, country: "US", flag: "🇺🇸",
    avatar: "/images/learners/marcus-us.jpg",
    bio: "High school student, basketball player",
    style: "curious", ageGroup: "teen" },
  
  { id: "sarah-ca", name: "Sarah", age: 45, country: "CA", flag: "🇨🇦",
    avatar: "/images/learners/sarah-ca.jpg",
    bio: "Teacher, learns with her students",
    style: "supportive", ageGroup: "adult" },
  
  { id: "grandpa-joe", name: "Joe", age: 72, country: "US", flag: "🇺🇸",
    avatar: "/images/learners/joe-us.jpg",
    bio: "Retired engineer, lifelong learner",
    style: "wise", ageGroup: "senior" },

  { id: "maya-mx", name: "Maya", age: 34, country: "MX", flag: "🇲🇽",
    avatar: "/images/learners/maya-mx.jpg",
    bio: "Architect, loves connecting ideas",
    style: "creative", ageGroup: "adult" },

  // === EUROPE ===
  { id: "james-uk", name: "James", age: 31, country: "GB", flag: "🇬🇧",
    avatar: "/images/learners/james-uk.jpg",
    bio: "Data analyst, morning learner",
    style: "methodical", ageGroup: "adult" },
  
  { id: "charlotte-uk", name: "Charlotte", age: 8, country: "GB", flag: "🇬🇧",
    avatar: "/images/learners/charlotte-uk.jpg",
    bio: "Loves dinosaurs and space",
    style: "wonder", ageGroup: "child" },
  
  { id: "marie-fr", name: "Marie", age: 52, country: "FR", flag: "🇫🇷",
    avatar: "/images/learners/marie-fr.jpg",
    bio: "Museum curator, art lover",
    style: "reflective", ageGroup: "adult" },
  
  { id: "lucas-fr", name: "Lucas", age: 19, country: "FR", flag: "🇫🇷",
    avatar: "/images/learners/lucas-fr.jpg",
    bio: "University student, philosophy major",
    style: "questioning", ageGroup: "young-adult" },
  
  { id: "hans-de", name: "Hans", age: 67, country: "DE", flag: "🇩🇪",
    avatar: "/images/learners/hans-de.jpg",
    bio: "Retired professor, still curious",
    style: "scholarly", ageGroup: "senior" },
  
  { id: "lena-de", name: "Lena", age: 24, country: "DE", flag: "🇩🇪",
    avatar: "/images/learners/lena-de.jpg",
    bio: "Medical student, studies at night",
    style: "dedicated", ageGroup: "young-adult" },
  
  { id: "isabella-it", name: "Isabella", age: 38, country: "IT", flag: "🇮🇹",
    avatar: "/images/learners/isabella-it.jpg",
    bio: "Chef, connects food to history",
    style: "passionate", ageGroup: "adult" },
  
  { id: "sven-se", name: "Sven", age: 29, country: "SE", flag: "🇸🇪",
    avatar: "/images/learners/sven-se.jpg",
    bio: "Product designer, visual learner",
    style: "creative", ageGroup: "adult" },
  
  { id: "nina-no", name: "Nina", age: 41, country: "NO", flag: "🇳🇴",
    avatar: "/images/learners/nina-no.jpg",
    bio: "Marine biologist, nature lover",
    style: "scientific", ageGroup: "adult" },
  
  { id: "olga-ua", name: "Olga", age: 33, country: "UA", flag: "🇺🇦",
    avatar: "/images/learners/olga-ua.jpg",
    bio: "Programmer, learns while commuting",
    style: "efficient", ageGroup: "adult" },

  // === ASIA ===
  { id: "yuki-jp", name: "Yuki", age: 26, country: "JP", flag: "🇯🇵",
    avatar: "/images/learners/yuki-jp.jpg",
    bio: "Graphic designer, anime fan",
    style: "artistic", ageGroup: "young-adult" },
  
  { id: "haruto-jp", name: "Haruto", age: 12, country: "JP", flag: "🇯🇵",
    avatar: "/images/learners/haruto-jp.jpg",
    bio: "Middle schooler, loves science",
    style: "curious", ageGroup: "child" },
  
  { id: "sakura-jp", name: "Sakura", age: 58, country: "JP", flag: "🇯🇵",
    avatar: "/images/learners/sakura-jp.jpg",
    bio: "Tea ceremony teacher, mindful learner",
    style: "contemplative", ageGroup: "adult" },
  
  { id: "priya-in", name: "Priya", age: 22, country: "IN", flag: "🇮🇳",
    avatar: "/images/learners/priya-in.jpg",
    bio: "Engineering student, ambitious",
    style: "driven", ageGroup: "young-adult" },
  
  { id: "arjun-in", name: "Arjun", age: 35, country: "IN", flag: "🇮🇳",
    avatar: "/images/learners/arjun-in.jpg",
    bio: "Doctor, learns with his children",
    style: "nurturing", ageGroup: "adult" },
  
  { id: "ananya-in", name: "Ananya", age: 9, country: "IN", flag: "🇮🇳",
    avatar: "/images/learners/ananya-in.jpg",
    bio: "Loves drawing and stories",
    style: "imaginative", ageGroup: "child" },
  
  { id: "wei-cn", name: "Wei", age: 44, country: "CN", flag: "🇨🇳",
    avatar: "/images/learners/wei-cn.jpg",
    bio: "Business owner, practical learner",
    style: "pragmatic", ageGroup: "adult" },
  
  { id: "mei-cn", name: "Mei", age: 17, country: "CN", flag: "🇨🇳",
    avatar: "/images/learners/mei-cn.jpg",
    bio: "High school senior, exam prep",
    style: "focused", ageGroup: "teen" },
  
  { id: "jin-kr", name: "Jin", age: 27, country: "KR", flag: "🇰🇷",
    avatar: "/images/learners/jin-kr.jpg",
    bio: "Game developer, night owl",
    style: "creative", ageGroup: "young-adult" },
  
  { id: "soo-yeon-kr", name: "Soo-yeon", age: 63, country: "KR", flag: "🇰🇷",
    avatar: "/images/learners/soo-yeon-kr.jpg",
    bio: "Grandmother, learning with grandkids",
    style: "patient", ageGroup: "senior" },

  // === MIDDLE EAST ===
  { id: "ahmed-eg", name: "Ahmed", age: 30, country: "EG", flag: "🇪🇬",
    avatar: "/images/learners/ahmed-eg.jpg",
    bio: "History teacher, ancient cultures",
    style: "storyteller", ageGroup: "adult" },
  
  { id: "fatima-eg", name: "Fatima", age: 21, country: "EG", flag: "🇪🇬",
    avatar: "/images/learners/fatima-eg.jpg",
    bio: "Journalism student, curious",
    style: "investigative", ageGroup: "young-adult" },
  
  { id: "omar-ae", name: "Omar", age: 39, country: "AE", flag: "🇦🇪",
    avatar: "/images/learners/omar-ae.jpg",
    bio: "Entrepreneur, busy schedule",
    style: "efficient", ageGroup: "adult" },
  
  { id: "layla-ae", name: "Layla", age: 14, country: "AE", flag: "🇦🇪",
    avatar: "/images/learners/layla-ae.jpg",
    bio: "Aspiring scientist, robotics club",
    style: "experimental", ageGroup: "teen" },

  // === AFRICA ===
  { id: "kofi-gh", name: "Kofi", age: 25, country: "GH", flag: "🇬🇭",
    avatar: "/images/learners/kofi-gh.jpg",
    bio: "Agricultural engineer, community builder",
    style: "practical", ageGroup: "young-adult" },
  
  { id: "ama-gh", name: "Ama", age: 48, country: "GH", flag: "🇬🇭",
    avatar: "/images/learners/ama-gh.jpg",
    bio: "School principal, lifelong educator",
    style: "mentoring", ageGroup: "adult" },
  
  { id: "aisha-ke", name: "Aisha", age: 20, country: "KE", flag: "🇰🇪",
    avatar: "/images/learners/aisha-ke.jpg",
    bio: "Wildlife conservation student",
    style: "passionate", ageGroup: "young-adult" },
  
  { id: "thabo-za", name: "Thabo", age: 36, country: "ZA", flag: "🇿🇦",
    avatar: "/images/learners/thabo-za.jpg",
    bio: "Jazz musician, creative thinker",
    style: "artistic", ageGroup: "adult" },
  
  { id: "naledi-za", name: "Naledi", age: 11, country: "ZA", flag: "🇿🇦",
    avatar: "/images/learners/naledi-za.jpg",
    bio: "Loves math puzzles and soccer",
    style: "playful", ageGroup: "child" },
  
  { id: "adebayo-ng", name: "Adebayo", age: 42, country: "NG", flag: "🇳🇬",
    avatar: "/images/learners/adebayo-ng.jpg",
    bio: "Tech entrepreneur, Lagos",
    style: "innovative", ageGroup: "adult" },

  // === SOUTH AMERICA ===
  { id: "maria-br", name: "Maria", age: 28, country: "BR", flag: "🇧🇷",
    avatar: "/images/learners/maria-br.jpg",
    bio: "Nurse, learns between shifts",
    style: "compassionate", ageGroup: "adult" },
  
  { id: "pedro-br", name: "Pedro", age: 55, country: "BR", flag: "🇧🇷",
    avatar: "/images/learners/pedro-br.jpg",
    bio: "Fisherman, loves ocean science",
    style: "experiential", ageGroup: "adult" },
  
  { id: "carlos-ar", name: "Carlos", age: 32, country: "AR", flag: "🇦🇷",
    avatar: "/images/learners/carlos-ar.jpg",
    bio: "Psychologist, interested in behavior",
    style: "analytical", ageGroup: "adult" },
  
  { id: "diego-cl", name: "Diego", age: 18, country: "CL", flag: "🇨🇱",
    avatar: "/images/learners/diego-cl.jpg",
    bio: "Astronomy enthusiast, stargazer",
    style: "wonder", ageGroup: "young-adult" },
  
  { id: "valentina-co", name: "Valentina", age: 7, country: "CO", flag: "🇨🇴",
    avatar: "/images/learners/valentina-co.jpg",
    bio: "Loves animals and colors",
    style: "playful", ageGroup: "child" },

  // === OCEANIA ===
  { id: "lisa-au", name: "Lisa", age: 37, country: "AU", flag: "🇦🇺",
    avatar: "/images/learners/lisa-au.jpg",
    bio: "Environmental scientist, beach walks",
    style: "observant", ageGroup: "adult" },
  
  { id: "jack-nz", name: "Jack", age: 23, country: "NZ", flag: "🇳🇿",
    avatar: "/images/learners/jack-nz.jpg",
    bio: "Outdoor guide, nature lover",
    style: "adventurous", ageGroup: "young-adult" },

  // === SOUTHEAST ASIA ===
  { id: "linh-vn", name: "Linh", age: 29, country: "VN", flag: "🇻🇳",
    avatar: "/images/learners/linh-vn.jpg",
    bio: "Coffee shop owner, morning routine",
    style: "steady", ageGroup: "adult" },
  
  { id: "ling-sg", name: "Ling", age: 45, country: "SG", flag: "🇸🇬",
    avatar: "/images/learners/ling-sg.jpg",
    bio: "Finance executive, efficient learner",
    style: "structured", ageGroup: "adult" },
  
  { id: "kai-th", name: "Kai", age: 19, country: "TH", flag: "🇹🇭",
    avatar: "/images/learners/kai-th.jpg",
    bio: "University student, travel lover",
    style: "open-minded", ageGroup: "young-adult" },
  
  { id: "putri-id", name: "Putri", age: 31, country: "ID", flag: "🇮🇩",
    avatar: "/images/learners/putri-id.jpg",
    bio: "Teacher, passionate about education",
    style: "nurturing", ageGroup: "adult" },

  // === ADDITIONAL DIVERSE PERSONAS ===
  { id: "zara-pk", name: "Zara", age: 26, country: "PK", flag: "🇵🇰",
    avatar: "/images/learners/zara-pk.jpg",
    bio: "Social worker, community focus",
    style: "empathetic", ageGroup: "young-adult" },
  
  { id: "elena-ru", name: "Elena", age: 40, country: "RU", flag: "🇷🇺",
    avatar: "/images/learners/elena-ru.jpg",
    bio: "Ballet teacher, disciplined",
    style: "precise", ageGroup: "adult" },
  
  { id: "tomasz-pl", name: "Tomasz", age: 50, country: "PL", flag: "🇵🇱",
    avatar: "/images/learners/tomasz-pl.jpg",
    bio: "Carpenter, hands-on learner",
    style: "practical", ageGroup: "adult" },
  
  { id: "anna-gr", name: "Anna", age: 65, country: "GR", flag: "🇬🇷",
    avatar: "/images/learners/anna-gr.jpg",
    bio: "Retired teacher, mythology lover",
    style: "storytelling", ageGroup: "senior" },
  
  { id: "chen-tw", name: "Chen", age: 34, country: "TW", flag: "🇹🇼",
    avatar: "/images/learners/chen-tw.jpg",
    bio: "Chip designer, tech enthusiast",
    style: "technical", ageGroup: "adult" },
  
  { id: "fatou-sn", name: "Fatou", age: 22, country: "SN", flag: "🇸🇳",
    avatar: "/images/learners/fatou-sn.jpg",
    bio: "Medical student, community health",
    style: "dedicated", ageGroup: "young-adult" },
  
  { id: "miguel-es", name: "Miguel", age: 47, country: "ES", flag: "🇪🇸",
    avatar: "/images/learners/miguel-es.jpg",
    bio: "Chef, culinary arts lover",
    style: "sensory", ageGroup: "adult" },
  
  { id: "ana-pt", name: "Ana", age: 15, country: "PT", flag: "🇵🇹",
    avatar: "/images/learners/ana-pt.jpg",
    bio: "Surfer, ocean science fan",
    style: "active", ageGroup: "teen" },
];

// ═══════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/**
 * Get a random persona, optionally filtered
 */
function getRandomPersona(filters = {}) {
  let pool = LEARNER_PERSONAS;
  
  if (filters.ageGroup) {
    pool = pool.filter(p => p.ageGroup === filters.ageGroup);
  }
  if (filters.style) {
    pool = pool.filter(p => p.style === filters.style);
  }
  if (filters.exclude) {
    pool = pool.filter(p => !filters.exclude.includes(p.id));
  }
  
  return pool[Math.floor(Math.random() * pool.length)] || LEARNER_PERSONAS[0];
}

/**
 * Get multiple unique personas
 */
function getUniquePersonas(count, filters = {}) {
  const personas = [];
  const usedIds = new Set();
  
  for (let i = 0; i < count && personas.length < LEARNER_PERSONAS.length; i++) {
    const persona = getRandomPersona({ ...filters, exclude: Array.from(usedIds) });
    if (!usedIds.has(persona.id)) {
      personas.push(persona);
      usedIds.add(persona.id);
    }
  }
  
  return personas;
}

/**
 * Get persona by ID
 */
function getPersonaById(id) {
  return LEARNER_PERSONAS.find(p => p.id === id);
}

/**
 * Get avatar URL with fallback
 */
function getAvatarUrl(persona) {
  // Return flag as fallback if no image exists
  return persona.avatar || null;
}

/**
 * Format persona for display
 */
function formatPersonaDisplay(persona) {
  return {
    name: persona.name,
    flag: persona.flag,
    avatar: getAvatarUrl(persona),
    ageDisplay: formatAge(persona.age),
  };
}

function formatAge(age) {
  if (age < 13) return null; // Don't show age for children
  if (age < 20) return 'teen';
  if (age < 30) return '20s';
  if (age < 40) return '30s';
  if (age < 50) return '40s';
  if (age < 60) return '50s';
  if (age < 70) return '60s';
  return '70+';
}

// ═══════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════

window.LEARNER_PERSONAS = LEARNER_PERSONAS;
window.getRandomPersona = getRandomPersona;
window.getUniquePersonas = getUniquePersonas;
window.getPersonaById = getPersonaById;
window.formatPersonaDisplay = formatPersonaDisplay;

