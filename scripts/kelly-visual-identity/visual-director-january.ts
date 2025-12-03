/**
 * VISUAL DIRECTOR - January Full Batch (Lessons 001-031)
 * 
 * Input: Full January Lesson List
 * Output: visual-manifest-january.json
 */

import * as fs from "fs";
import * as path from "path";

// === JANUARY LESSONS (001-031) ===
const JANUARY_LESSONS = [
  { day: 1, title: "Starting Fresh" },
  { day: 2, title: "The Three Lives of Water" },
  { day: 3, title: "Where Clouds Come From" },
  { day: 4, title: "How Light Travels" },
  { day: 5, title: "How Sound Moves" },
  { day: 6, title: "What's Inside a Seed" },
  { day: 7, title: "What Stars Are Made Of" },
  { day: 8, title: "What Makes a Real Friend" },
  { day: 9, title: "How Kindness Spreads" },
  { day: 10, title: "The Art of Really Listening" },
  { day: 11, title: "Why Patience Pays Off" },
  { day: 12, title: "How Gratitude Changes You" },
  { day: 13, title: "Why We Dream" },
  { day: 14, title: "The Power of Questions" },
  { day: 15, title: "How Your Brain Learns" },
  { day: 16, title: "What Makes Music Feel" },
  { day: 17, title: "Why Mistakes Matter" },
  { day: 18, title: "How Plants Eat Sunlight" },
  { day: 19, title: "The Story in Your Blood" },
  { day: 20, title: "Why We Need Sleep" },
  { day: 21, title: "How to Disagree Well" },
  { day: 22, title: "What Gravity Really Does" },
  { day: 23, title: "The Gift of Boredom" },
  { day: 24, title: "How Memory Works" },
  { day: 25, title: "Why Stories Change Us" },
  { day: 26, title: "The Science of Smiling" },
  { day: 27, title: "How Ice Shapes Land" },
  { day: 28, title: "What Courage Really Means" },
  { day: 29, title: "The Hidden Life of Soil" },
  { day: 30, title: "Why We Help Strangers" },
  { day: 31, title: "How Change Happens" }
];

// === THE CREATIVE ENGINE ===
function dreamUpVisuals(lesson: { day: number; title: string }) {
  
  const title = lesson.title.toLowerCase();
  
  // Heuristic Logic for Scenes/Props (Mocking LLM Creativity)
  let scene = "clean minimal studio with soft abstract shapes";
  let mood = "soft daylight, approachable";
  let propAction = "holding a notebook in left hand, right hand relaxed";
  
  // SCIENCE / NATURE
  if (title.includes("water") || title.includes("ice") || title.includes("cloud")) {
    scene = "ethereal landscape with floating water droplets and mist, soft blue tones";
    mood = "fresh, clean, aquatic lighting";
    propAction = "holding a clear glass sphere resembling a water drop in right hand, looking at it";
  }
  else if (title.includes("light") || title.includes("star") || title.includes("gravity")) {
    scene = "cosmic environment with deep purple nebula and glowing stars, cinematic depth";
    mood = "mysterious, dramatic rim lighting, scientific wonder";
    propAction = "holding a glowing prism or orb in right hand, illuminating face slightly";
  }
  else if (title.includes("plant") || title.includes("seed") || title.includes("soil") || title.includes("grow")) {
    scene = "lush botanical garden with oversized leaves and dappled sunlight, macro texture";
    mood = "warm, organic, sun-drenched green tones";
    propAction = "holding a small potted seedling in both hands carefully";
  }
  else if (title.includes("blood") || title.includes("brain") || title.includes("sleep") || title.includes("memory")) {
    scene = "abstract biological visualization, soft floating cells or neural networks, clean white and red/pink accents";
    mood = "clinical but warm, soft focus, educational";
    propAction = "pointing to a floating holographic diagram of a cell";
  }
  
  // SOCIAL / EMOTIONAL
  else if (title.includes("friend") || title.includes("kindness") || title.includes("listen") || title.includes("help")) {
    scene = "warm, inviting living room space with soft textiles and golden light";
    mood = "cozy, empathetic, soft amber and magenta tones";
    propAction = "hand on heart gesture, sincere expression";
  }
  
  // GROWTH / MINDSET
  else if (title.includes("fresh") || title.includes("change") || title.includes("courage") || title.includes("mistakes")) {
    scene = "open horizon at sunrise, path leading forward, vast sky";
    mood = "hopeful, golden hour, inspiring, high contrast";
    propAction = "looking toward horizon, hands on hips in confident stance";
  }
  
  // CREATIVE
  else if (title.includes("music") || title.includes("stories") || title.includes("questions")) {
    scene = "abstract creative studio with floating geometric shapes and color swatches";
    mood = "vibrant, energetic, colorful lighting";
    propAction = "holding a vintage microphone or book, expressive gesture";
  }

  // BASE: Kelly Anchor
  const KELLY = "kelly, young woman, light blue sweater, blue jeans, white sneakers";
  const STYLE = "photorealistic editorial photography, cinematic lighting, 8k, high fidelity, soft shadows";

  return {
    lesson_id: lesson.day,
    title: lesson.title,
    assets: [
      // 1. HERO (Thumbnail)
      {
        type: "hero",
        filename: `lesson-${lesson.day}-hero.png`,
        prompt: `${KELLY}, walking mid-stride profile view through ${scene}, full body shot including feet, wide shot, looking ahead confidently, ${mood}, ${STYLE}`
      },
      // 2. GUIDE (Portrait)
      {
        type: "guide_point",
        filename: `lesson-${lesson.day}-guide-point.png`,
        prompt: `${KELLY}, medium shot waist up facing camera, ${scene}, right hand pointing upwards with index finger, left hand resting on imaginary surface, making eye contact, ${mood}, ${STYLE}, vertical 9:16 aspect ratio`
      },
      // 3. REACTION (Square)
      {
        type: "reaction",
        filename: `lesson-${lesson.day}-reaction.png`,
        prompt: `${KELLY}, close up portrait, face only, ${scene}, expression of curiosity and wonder, looking slightly off camera, ${mood}, ${STYLE}, square 1:1 aspect ratio`
      },
      // 4. PROP (Action)
      {
        type: "prop",
        filename: `lesson-${lesson.day}-prop.png`,
        prompt: `${KELLY}, medium shot waist up, ${scene}, ${propAction}, ${mood}, ${STYLE}`
      },
      // 5. BACKGROUND (Clean)
      {
        type: "background",
        filename: `lesson-${lesson.day}-bg.png`,
        prompt: `${scene}, empty scene, background only, no people, ${mood}, ${STYLE}, wide shot`
      }
    ]
  };
}

async function main() {
  console.log("🧠 VISUAL DIRECTOR: Generating January Manifest...");
  
  const manifest = JANUARY_LESSONS.map(lesson => dreamUpVisuals(lesson));
  
  const outputPath = path.join(process.cwd(), "scripts", "kelly-visual-identity", "visual-manifest-january.json");
  fs.writeFileSync(outputPath, JSON.stringify(manifest, null, 2));
  
  console.log(`✨ Manifest created with ${manifest.length} lessons.`);
  console.log(`📂 Saved to: ${outputPath}`);
}

main().catch(console.error);




