/**
 * Fix 11 Critically Broken Lessons
 * Regenerate proper extended_explanation content for systematically corrupted lessons
 */

import { pipeline } from '@xenova/transformers';

// Proper content for each broken lesson
const PROPER_CONTENT = {
  58: {
    topic: "Life in the Desert",
    extended_explanation: "Deserts are extreme environments characterized by very low rainfall (typically less than 10 inches per year) and dramatic temperature fluctuations. Despite these harsh conditions, deserts are home to remarkably adapted life forms. Desert plants like cacti store water in their thick stems and have spines instead of leaves to reduce water loss. Animals such as camels, fennec foxes, and roadrunners have evolved unique strategies: some are nocturnal to avoid daytime heat, others can survive without drinking water by extracting moisture from food, and many have specialized kidneys that conserve water. Desert ecosystems demonstrate that life finds a way even in the most challenging conditions, with organisms developing incredible adaptations for survival."
  },
  61: {
    topic: "The Power of Grass",
    extended_explanation: "Grasses are among the most successful plants on Earth, covering approximately 40% of the planet's land surface (excluding Antarctica). The grass family (Poaceae) includes over 10,000 species, from tiny lawn grasses to towering bamboo. What makes grass so powerful is its unique growth pattern: unlike most plants that grow from their tips, grasses grow from their base, allowing them to survive grazing, mowing, and trampling. This resilience has made grass the foundation of major ecosystems (prairies, savannas, steppes) and human civilization itself. Wheat, rice, corn, barley, and oats—the crops that feed most of humanity—are all grasses. Grass also prevents soil erosion, sequesters carbon, and provides habitat for countless species. Without grass, Earth's ecosystems and human agriculture would collapse."
  },
  64: {
    topic: "Worlds Without Light",
    extended_explanation: "The deep ocean, below about 1,000 meters (3,280 feet), exists in perpetual darkness. Sunlight cannot penetrate these depths, creating an alien world unlike any environment on Earth's surface. Yet this lightless realm teems with life that has evolved extraordinary adaptations. Many deep-sea creatures produce their own light through bioluminescence—chemical reactions that create glowing lures, camouflage, or communication signals. The anglerfish dangles a glowing lure to attract prey. The vampire squid flashes bioluminescent clouds to confuse predators. Some organisms have enormous eyes to capture any trace of light, while others have no eyes at all, relying on other senses. The deep ocean is Earth's largest habitat, and we've explored less than 5% of it. These lightless worlds remind us that life adapts to even the most extreme conditions."
  },
  65: {
    topic: "How Islands Are Born",
    extended_explanation: "Islands form through several fascinating geological processes. Volcanic islands, like Hawaii, are born when underwater volcanoes erupt repeatedly over millions of years, building mountains from the ocean floor until they break the surface. The Hawaiian Islands formed as the Pacific Plate moved over a stationary 'hot spot' in Earth's mantle, creating a chain of islands. Continental islands, like the British Isles, were once connected to continents but became separated by rising sea levels or tectonic shifts. Coral atolls begin as volcanic islands that slowly sink while coral reefs grow around them, eventually leaving only a ring of coral. Barrier islands form along coastlines from accumulated sediment. Each island is a unique ecosystem, often hosting species found nowhere else on Earth. Islands are natural laboratories for studying evolution and adaptation."
  },
  114: {
    topic: "Lifting Heavy Things Easily",
    extended_explanation: "Pulleys are simple machines that make lifting heavy objects easier by redirecting force and, in some configurations, multiplying it. A single fixed pulley (like a flagpole pulley) simply changes the direction of force—you pull down to lift something up—but doesn't reduce the force needed. However, a movable pulley, where the pulley moves with the load, provides a mechanical advantage: it cuts the force needed in half. By combining multiple pulleys in a block-and-tackle system, you can multiply force dramatically. With four pulleys, you need only one-fourth the force to lift an object, though you must pull four times as much rope. This trade-off between force and distance is fundamental to all simple machines. Pulleys have been used for thousands of years in construction, sailing, and industry. Understanding pulleys reveals the elegant physics principle that you can't get something for nothing—you trade force for distance."
  },
  122: {
    topic: "How Movies Create Motion",
    extended_explanation: "Movies create the illusion of motion through a phenomenon called persistence of vision and the phi phenomenon. When still images are displayed in rapid succession—typically 24 frames per second in traditional film or 30-60 fps in digital video—our brains perceive continuous motion rather than separate images. This happens because our visual system retains an image for a fraction of a second after it disappears, and our brain fills in the gaps between frames. Early motion pictures in the 1890s proved this principle, and it remains the foundation of all film and video today. Each frame is a still photograph, but shown fast enough, they blend into seamless movement. This same principle powers animation, where artists draw or generate each frame individually. Modern digital cinema can use higher frame rates (48, 60, or even 120 fps) for smoother motion, but 24 fps remains the standard because it provides a cinematic 'look' audiences expect. Movies are literally thousands of lies per second that add up to a truth our brains believe."
  },
  241: {
    topic: "How Plants Eat Sunlight",
    extended_explanation: "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in sugar molecules. This process occurs primarily in the chloroplasts of plant cells, which contain chlorophyll—the green pigment that captures light energy. The chemical equation is deceptively simple: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. Plants take in carbon dioxide from the air and water from the soil, use light energy to break apart these molecules, and reassemble them into glucose (sugar) and oxygen. The oxygen is released as a waste product—which is fortunate for us, as photosynthesis produces most of Earth's breathable oxygen. The glucose becomes the building block for all plant growth: cellulose for cell walls, starches for energy storage, and the foundation of the plant's structure. Photosynthesis is arguably the most important chemical reaction on Earth, as it forms the base of nearly all food chains and maintains our atmosphere."
  },
  242: {
    topic: "How Bodies Make Energy",
    extended_explanation: "Cellular respiration is the process by which living cells convert nutrients (primarily glucose) into usable energy in the form of ATP (adenosine triphosphate). This process occurs in the mitochondria—the 'powerhouses' of cells. The chemical equation mirrors photosynthesis in reverse: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + energy (ATP). Your body breaks down the food you eat into glucose, which enters cells and undergoes a series of chemical reactions. Glycolysis breaks glucose into smaller molecules, the citric acid cycle extracts electrons, and the electron transport chain uses those electrons to generate ATP. A single glucose molecule yields about 36-38 ATP molecules. This ATP powers everything your body does: muscle contractions, nerve signals, protein synthesis, cell division, and maintaining body temperature. You produce and use roughly your body weight in ATP every day. Metabolism is the sum of all these chemical reactions that keep you alive."
  },
  245: {
    topic: "Getting Rid of Waste",
    extended_explanation: "Waste removal is essential for life at every scale—from individual cells to entire ecosystems. In living organisms, cells produce metabolic waste products that would be toxic if allowed to accumulate. Humans have multiple waste removal systems: the kidneys filter blood and produce urine to eliminate nitrogen waste and excess water; the liver processes toxins and produces bile; the lungs expel carbon dioxide; the skin releases waste through sweat; and the digestive system eliminates solid waste. At the ecosystem level, decomposers (bacteria, fungi, and detritivores) break down dead organisms and waste, recycling nutrients back into the environment. Without decomposition, dead material would accumulate and nutrients would remain locked away, halting life cycles. Human societies face waste management challenges: sewage treatment, garbage disposal, recycling, and pollution control. The principle is universal: waste is simply matter or energy in the wrong place. Effective waste management—whether in a cell, organism, or city—is crucial for health and sustainability."
  },
  277: {
    topic: "Power From Splitting Atoms",
    extended_explanation: "Nuclear energy is released when the nucleus of an atom is split (fission) or combined (fusion). Nuclear fission, used in power plants, involves splitting heavy atoms like uranium-235 or plutonium-239. When a neutron strikes a uranium nucleus, it splits into two smaller atoms, releases 2-3 more neutrons, and converts a tiny amount of mass into enormous energy (E=mc²). Those released neutrons can split other atoms, creating a chain reaction. In a controlled reactor, this chain reaction is carefully managed to produce steady heat, which boils water to spin turbines and generate electricity. A single kilogram of uranium-235 can produce as much energy as burning 3 million kilograms of coal. Nuclear power generates about 10% of the world's electricity with zero carbon emissions during operation. However, it produces radioactive waste that remains dangerous for thousands of years and carries risks of accidents. Nuclear fusion—combining light atoms like hydrogen—powers the sun and promises even cleaner energy, but remains technologically challenging on Earth."
  },
  364: {
    topic: "Starting Fresh",
    extended_explanation: "Fresh starts are more than just calendar flips—they're psychological powerhouses that give us permission to change. Researchers call this the 'fresh start effect': people are significantly more likely to pursue goals after temporal landmarks like New Year's Day, birthdays, Mondays, or the first day of a month. These dates create mental chapters in our life story, allowing us to psychologically distance ourselves from past failures and imagine ourselves as different people. Studies show gym visits spike on Mondays, searches for 'diet' jump 80% on January 1st, and people are 62% more likely to pursue goals after meaningful dates. The effect works because it helps us separate 'old me' from 'new me,' making past mistakes feel like they belong to someone else. But here's the secret: you don't need to wait for January 1st. Any moment can be a fresh start if you decide it is. The calendar doesn't give you permission to change—you do. Every sunrise, every breath, every moment you choose to try again is a fresh start. The power isn't in the date; it's in your decision to begin again."
  }
};

console.log('\n═══════════════════════════════════════════════════════════════');
console.log('   🔧 FIXING 11 CRITICALLY BROKEN LESSONS');
console.log('   Replacing wrong content with proper extended_explanation');
console.log('═══════════════════════════════════════════════════════════════\n');

// Output the SQL update statements
for (const [day, content] of Object.entries(PROPER_CONTENT)) {
  console.log(`-- Day ${day}: ${content.topic}`);
  console.log(`UPDATE core_lessons`);
  console.log(`SET extended_explanation = '${content.extended_explanation.replace(/'/g, "''")}'`);
  console.log(`WHERE day_number = ${day};`);
  console.log();
}

console.log('\n═══════════════════════════════════════════════════════════════');
console.log('   ✅ SQL STATEMENTS GENERATED');
console.log('   Copy and run these in Supabase SQL editor');
console.log('═══════════════════════════════════════════════════════════════\n');



