-- CURRICULUM DATA FIX SCRIPT
-- Generated 2026-02-04
-- Fixes icon_emoji misalignments and duplicate universal_truths

-- =========================================
-- PART 1: FIX ICON EMOJIS (Days 57-89+)
-- =========================================

-- Day 57: "Where Lakes Come From" - has 🌵 (cactus), should be lake-related
UPDATE core_lessons SET icon_emoji = '🏔️' WHERE track = 'learn' AND day_number = 57;

-- Day 58: "Life in the Desert" - has 🌲 (tree), should be 🌵 (cactus)
UPDATE core_lessons SET icon_emoji = '🌵' WHERE track = 'learn' AND day_number = 58;

-- Day 59: "The Secret Life of Forests" - has 🐠 (fish), should be 🌲 (tree)
UPDATE core_lessons SET icon_emoji = '🌲' WHERE track = 'learn' AND day_number = 59;

-- Day 60: "Why Jungles Are So Alive" - has ⛏️ (pickaxe), should be 🌴 (palm)
UPDATE core_lessons SET icon_emoji = '🌴' WHERE track = 'learn' AND day_number = 60;

-- Day 61: "The Power of Grass" - has 🦋 (butterfly), should be 🌾 (grass)
UPDATE core_lessons SET icon_emoji = '🌾' WHERE track = 'learn' AND day_number = 61;

-- Day 62: "Why Wetlands Matter" - has 🐦, could keep as bird or use 🐸 (frog)
UPDATE core_lessons SET icon_emoji = '🐸' WHERE track = 'learn' AND day_number = 62;

-- Day 63: "Cities Under the Sea" - has 🐠, should be 🪸 (coral)
UPDATE core_lessons SET icon_emoji = '🪸' WHERE track = 'learn' AND day_number = 63;

-- Day 64: "Worlds Without Light" - has 🦁 (lion), should be 🦇 (bat/cave)
UPDATE core_lessons SET icon_emoji = '🦇' WHERE track = 'learn' AND day_number = 64;

-- Day 65: "How Islands Are Born" - has 🦎 (lizard), should be 🏝️ (island)
UPDATE core_lessons SET icon_emoji = '🏝️' WHERE track = 'learn' AND day_number = 65;

-- Day 66: "What's Living in the Dirt" - has 🐸 (frog), should be 🪱 (worm)
UPDATE core_lessons SET icon_emoji = '🪱' WHERE track = 'learn' AND day_number = 66;

-- Day 67: "The Stories Rocks Tell" - has 🌿 (plant), should be 🪨 (rock)
UPDATE core_lessons SET icon_emoji = '🪨' WHERE track = 'learn' AND day_number = 67;

-- Day 68: "Earth's Hidden Treasures" - has 🌳 (tree), should be 💎 (gem)
UPDATE core_lessons SET icon_emoji = '💎' WHERE track = 'learn' AND day_number = 68;

-- Day 69: "How Gems Are Made" - has 🌸 (flower), should be 💍 (gem ring)
UPDATE core_lessons SET icon_emoji = '💍' WHERE track = 'learn' AND day_number = 69;

-- Day 70: "Where Metals Come From" - has 🍄 (mushroom), should be ⚙️ (gear/metal)
UPDATE core_lessons SET icon_emoji = '⚙️' WHERE track = 'learn' AND day_number = 70;

-- Day 71: "What's In the Air You Breathe" - has 🦠 (germ), should be 🌬️ (wind/air)
UPDATE core_lessons SET icon_emoji = '🌬️' WHERE track = 'learn' AND day_number = 71;

-- Day 72: "Why We Need Oxygen" - has 🧬 (DNA), should be 🫁 (lungs)
UPDATE core_lessons SET icon_emoji = '🫁' WHERE track = 'learn' AND day_number = 72;

-- Day 73: "Carbon Is Everywhere" - has 🧠 (brain), should be ⚫ (carbon black)
UPDATE core_lessons SET icon_emoji = '⚫' WHERE track = 'learn' AND day_number = 73;

-- Day 74: "The Gas You Don't Notice" - has 🫀 (heart), should be 🌫️ (mist/gas)
UPDATE core_lessons SET icon_emoji = '🌫️' WHERE track = 'learn' AND day_number = 74;

-- Day 75: "The Simplest Element" - has 🦴 (bone), should be ⚛️ (atom)
UPDATE core_lessons SET icon_emoji = '⚛️' WHERE track = 'learn' AND day_number = 75;

-- Day 76: "Building Blocks of Everything" - has 💪 (muscle), should be 🧱 (blocks)
UPDATE core_lessons SET icon_emoji = '🧱' WHERE track = 'learn' AND day_number = 76;

-- Day 77: "When Atoms Connect" - has 🩸 (blood), should be 🔗 (links)
UPDATE core_lessons SET icon_emoji = '🔗' WHERE track = 'learn' AND day_number = 77;

-- Day 78: "The Tiny Units of Life" - has 🧤 (glove), should be 🔬 (microscope)
UPDATE core_lessons SET icon_emoji = '🔬' WHERE track = 'learn' AND day_number = 78;

-- Day 79: "Your Body's Instruction Manual" - has 👁️ (eye), should be 🧬 (DNA)
UPDATE core_lessons SET icon_emoji = '🧬' WHERE track = 'learn' AND day_number = 79;

-- Day 80: "What Blood Does All Day" - has 👂 (ear), should be 🩸 (blood)
UPDATE core_lessons SET icon_emoji = '🩸' WHERE track = 'learn' AND day_number = 80;

-- Day 84: "How Your Eyes See" - has 😊 (smiley), should be 👁️ (eye)
UPDATE core_lessons SET icon_emoji = '👁️' WHERE track = 'learn' AND day_number = 84;

-- Day 85: "How Your Ears Work" - has 😊 (smiley), should be 👂 (ear)
UPDATE core_lessons SET icon_emoji = '👂' WHERE track = 'learn' AND day_number = 85;

-- Day 86: "Your Heart Never Stops" - has 😢 (crying), should be ❤️ (heart)
UPDATE core_lessons SET icon_emoji = '❤️' WHERE track = 'learn' AND day_number = 86;

-- Day 87: "What Your Brain Does" - has 😨 (scared), should be 🧠 (brain)
UPDATE core_lessons SET icon_emoji = '🧠' WHERE track = 'learn' AND day_number = 87;

-- Day 88: "How Lungs Breathe for You" - has 😡 (angry), should be 🫁 (lungs)
UPDATE core_lessons SET icon_emoji = '🫁' WHERE track = 'learn' AND day_number = 88;

-- Day 89: "Your Body's Framework" - has 🎁 (gift), should be 🦴 (bone)
UPDATE core_lessons SET icon_emoji = '🦴' WHERE track = 'learn' AND day_number = 89;

-- =========================================
-- PART 2: FIX DUPLICATE UNIVERSAL_TRUTHS
-- =========================================

-- Day 34: "How Magnets Work" has wrong universal_truth (about work/economics)
UPDATE core_lessons SET universal_truth = 'Magnets attract or repel each other due to the alignment of their atomic domains—invisible forces at a distance.' WHERE track = 'learn' AND day_number = 34;

-- Day 89: "Your Body's Framework" has wrong universal_truth (about work/economics)
UPDATE core_lessons SET universal_truth = 'Your skeleton provides structure, protects vital organs, and produces blood cells inside the bone marrow.' WHERE track = 'learn' AND day_number = 89;

-- =========================================
-- VERIFICATION QUERIES
-- =========================================

-- Check icons for Days 57-90
-- SELECT day_number, topic, icon_emoji FROM core_lessons WHERE track = 'learn' AND day_number BETWEEN 57 AND 90 ORDER BY day_number;

-- Check for remaining duplicate universal_truths
-- SELECT universal_truth, COUNT(*) as count FROM core_lessons WHERE track = 'learn' GROUP BY universal_truth HAVING COUNT(*) > 1;
