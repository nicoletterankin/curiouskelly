/**
 * VISUAL DIRECTOR - AI Prompt Engineer for Kelly (v2 - Refined)
 * 
 * Updates:
 * - Added "full body shot including feet" to Hero prompts
 * - Added "dissociated limb" control to Prop prompts (one hand active, one anchored)
 * - Improved lighting descriptions
 */

import * as fs from "fs";
import * as path from "path";

// === LESSON INPUTS (December Pilot) ===
const DECEMBER_LESSONS = [
  { day: 335, title: "The Story of Money" },
  { day: 336, title: "How Banks Actually Work" },
  { day: 337, title: "The Power of Compound Interest" },
  { day: 338, title: "Needs vs. Wants" },
  { day: 339, title: "How to Save for a Dream" },
];

// === THE CREATIVE ENGINE ===
function dreamUpVisuals(lesson: { day: number; title: string }) {
  
  const isMoneyTopic = lesson.title.toLowerCase().includes("money") || 
                       lesson.title.toLowerCase().includes("bank") ||
                       lesson.title.toLowerCase().includes("save") ||
                       lesson.title.toLowerCase().includes("interest");

  // BASE: Kelly Anchor
  const KELLY = "kelly, young woman, light blue sweater, blue jeans, white sneakers";
  const STYLE = "photorealistic editorial photography, cinematic lighting, 8k, high fidelity, soft shadows";
  const NEGATIVE = "cartoon, illustration, anime, painting, drawing, sketch, blurry, low quality, watermark, text, logo, extra limbs, extra fingers, deformed, distorted face, wrong outfit, different clothes, holding objects, sitting, standing still, bad anatomy, distorted body, cropped feet";

  // DYNAMIC: Scene & Mood
  let scene = "";
  let mood = "";
  let propAction = "";
  
  if (isMoneyTopic) {
    scene = "surreal abstract bank vault with floating golden particles, clean architectural lines, circular vault door elements";
    mood = "warm golden hour lighting, feeling of value and security, cinematic depth";
    // Explicitly describe ONE hand holding, OTHER hand at side
    propAction = "holding a large gold coin in right hand, left hand relaxed at side, looking at coin";
  } else {
    scene = "clean minimal studio with soft abstract shapes, floating geometric forms";
    mood = "soft daylight, approachable and clear, high key lighting";
    propAction = "holding a notebook in left hand, right hand gesturing slightly, looking at notebook";
  }

  return {
    lesson_id: lesson.day,
    title: lesson.title,
    assets: [
      // 1. HERO (Thumbnail) - Walking/Active
      // Added: "full body shot including feet"
      {
        type: "hero",
        filename: `lesson-${lesson.day}-hero.png`,
        prompt: `${KELLY}, walking mid-stride profile view through ${scene}, full body shot including feet, wide shot, looking ahead confidently, ${mood}, ${STYLE}`
      },
      // 2. GUIDE (Portrait) - Pointing/Engaging
      {
        type: "guide_point",
        filename: `lesson-${lesson.day}-guide-point.png`,
        prompt: `${KELLY}, medium shot waist up facing camera, ${scene}, right hand pointing upwards with index finger, left hand resting on imaginary surface, making eye contact, ${mood}, ${STYLE}, vertical 9:16 aspect ratio`
      },
      // 3. REACTION (Square) - Emotional
      {
        type: "reaction",
        filename: `lesson-${lesson.day}-reaction.png`,
        prompt: `${KELLY}, close up portrait, face only, ${scene}, expression of curiosity and wonder, looking slightly off camera, ${mood}, ${STYLE}, square 1:1 aspect ratio`
      },
      // 4. PROP (Action) - Holding Object
      // Added: Specific hand instructions
      {
        type: "prop",
        filename: `lesson-${lesson.day}-prop.png`,
        prompt: `${KELLY}, medium shot waist up, ${scene}, ${propAction}, ${mood}, ${STYLE}`
      },
      // 5. BACKGROUND (Clean) - No Kelly
      {
        type: "background",
        filename: `lesson-${lesson.day}-bg.png`,
        prompt: `${scene}, empty scene, background only, no people, ${mood}, ${STYLE}, wide shot`
      }
    ]
  };
}

async function main() {
  console.log("🧠 VISUAL DIRECTOR v2: Dreaming up concepts...");
  
  const manifest = DECEMBER_LESSONS.map(lesson => dreamUpVisuals(lesson));
  
  const outputPath = path.join(process.cwd(), "scripts", "kelly-visual-identity", "visual-manifest-december-v2.json");
  fs.writeFileSync(outputPath, JSON.stringify(manifest, null, 2));
  
  console.log(`✨ Manifest v2 created with ${manifest.length} lessons.`);
  console.log(`📂 Saved to: ${outputPath}`);
}

main().catch(console.error);
