/**
 * Emergency Lessons Extended - 30 days of hardcoded fallback content
 * 
 * This provides a safety net when:
 * - Supabase is down
 * - D1 mirror is unavailable
 * - Static JSON files don't exist for the day
 * 
 * THE LESSON ALWAYS PLAYS.
 */

window.EMERGENCY_LESSONS_EXTENDED = {
  // Week 1: Foundations
  1: {
    id: 'emergency-1',
    day_number: 1,
    topic: 'The Sun',
    marketing_headline: 'Our life-giving star',
    marketing_tagline: 'Discover the power above',
    universal_truth: 'Our star gives life to everything on Earth.',
    greeting: 'Good morning! Let\'s explore the incredible power of the Sun.',
    script: 'The Sun is 93 million miles away, yet its light reaches us in just 8 minutes. Every second, it converts 4 million tons of matter into pure energy. This one star makes all life on Earth possible.',
    hero_image_url: '/images/lessons/sun.png'
  },
  2: {
    id: 'emergency-2',
    day_number: 2,
    topic: 'Why the Sky is Blue',
    marketing_headline: 'The secret of scattered light',
    marketing_tagline: 'Look up and wonder',
    universal_truth: 'Light bends and scatters to paint our world.',
    greeting: 'Have you ever wondered why the sky is blue?',
    script: 'Sunlight contains all colors of the rainbow. When it hits our atmosphere, blue light scatters more than other colors because it travels in shorter waves. This scattered blue light is what we see when we look up.',
    hero_image_url: '/images/lessons/sky.png'
  },
  3: {
    id: 'emergency-3',
    day_number: 3,
    topic: 'How Seeds Grow',
    marketing_headline: 'The miracle of germination',
    marketing_tagline: 'Life finds a way',
    universal_truth: 'Every giant oak began as a tiny seed with potential.',
    greeting: 'Today we\'re planting seeds of knowledge!',
    script: 'Inside every seed is a tiny plant embryo waiting for the right conditions. When it gets water, warmth, and oxygen, it breaks through its shell and reaches for the light. This process is called germination.',
    hero_image_url: '/images/lessons/seeds.png'
  },
  4: {
    id: 'emergency-4',
    day_number: 4,
    topic: 'The Water Cycle',
    marketing_headline: 'Nature\'s eternal recycling',
    marketing_tagline: 'The same water, forever flowing',
    universal_truth: 'Water is never created or destroyed—only transformed.',
    greeting: 'The water you drink today might have been drunk by a dinosaur!',
    script: 'Water evaporates from oceans and lakes, rises as vapor, condenses into clouds, then falls as rain or snow. This cycle has been running for billions of years—the same water, over and over.',
    hero_image_url: '/images/lessons/water-cycle.png'
  },
  5: {
    id: 'emergency-5',
    day_number: 5,
    topic: 'Why We Sleep',
    marketing_headline: 'The power of rest',
    marketing_tagline: 'Your brain\'s nightly maintenance',
    universal_truth: 'Sleep is when your brain organizes everything you learned.',
    greeting: 'Ready to learn why sleep is your superpower?',
    script: 'While you sleep, your brain is incredibly busy. It strengthens memories, clears out toxins, and repairs cells. That\'s why a good night\'s sleep makes you feel refreshed and helps you learn better.',
    hero_image_url: '/images/lessons/sleep.png'
  },
  6: {
    id: 'emergency-6',
    day_number: 6,
    topic: 'How Birds Fly',
    marketing_headline: 'Masters of the sky',
    marketing_tagline: 'Engineering perfection',
    universal_truth: 'Nature solved flight millions of years before humans.',
    greeting: 'Let\'s soar into the science of flight!',
    script: 'Birds have hollow bones that make them light, powerful chest muscles that power their wings, and feathers shaped like airfoils. When they flap, their wings push air down and back, lifting them up and forward.',
    hero_image_url: '/images/lessons/birds.png'
  },
  7: {
    id: 'emergency-7',
    day_number: 7,
    topic: 'The Moon',
    marketing_headline: 'Our faithful companion',
    marketing_tagline: 'Earth\'s only natural satellite',
    universal_truth: 'The Moon has watched over Earth for 4.5 billion years.',
    greeting: 'Tonight, look up and see our closest neighbor in space!',
    script: 'The Moon is about 239,000 miles away. It controls our tides, stabilizes Earth\'s tilt, and reflects sunlight to give us moonlit nights. It\'s the only place beyond Earth where humans have walked.',
    hero_image_url: '/images/lessons/moon.png'
  },
  
  // Week 2: Body & Mind
  8: {
    id: 'emergency-8',
    day_number: 8,
    topic: 'The Heart',
    marketing_headline: 'Your tireless engine',
    marketing_tagline: 'Beating 100,000 times a day',
    universal_truth: 'Your heart never takes a break—it\'s always working for you.',
    greeting: 'Let\'s explore the most amazing pump in the world!',
    script: 'Your heart beats about 100,000 times every day, pumping blood through 60,000 miles of blood vessels. That\'s enough blood to fill a swimming pool every year!',
    hero_image_url: '/images/lessons/heart.png'
  },
  9: {
    id: 'emergency-9',
    day_number: 9,
    topic: 'The Brain',
    marketing_headline: 'Your command center',
    marketing_tagline: '86 billion neurons working together',
    universal_truth: 'Your brain is the most complex object in the known universe.',
    greeting: 'Ready to explore the most amazing organ in your body?',
    script: 'Your brain has 86 billion neurons, each connected to thousands of others. It uses 20% of your body\'s energy and can process information faster than any computer ever built.',
    hero_image_url: '/images/lessons/brain.png'
  },
  10: {
    id: 'emergency-10',
    day_number: 10,
    topic: 'How We See',
    marketing_headline: 'Light to vision',
    marketing_tagline: 'Your eyes capture the world',
    universal_truth: 'Your eyes capture light, but your brain creates the picture.',
    greeting: 'Let\'s see how seeing really works!',
    script: 'Light enters your eye through the pupil, gets focused by the lens onto the retina at the back. Special cells called rods and cones convert light into electrical signals that travel to your brain.',
    hero_image_url: '/images/lessons/eyes.png'
  },
  
  // Week 3: Nature
  11: {
    id: 'emergency-11',
    day_number: 11,
    topic: 'Rainbows',
    marketing_headline: 'Light\'s colorful secret',
    marketing_tagline: 'When rain meets sun',
    universal_truth: 'White light contains all the colors of the rainbow.',
    greeting: 'Let\'s chase rainbows together!',
    script: 'Rainbows appear when sunlight passes through water droplets. The droplets act like tiny prisms, splitting white light into its seven colors: red, orange, yellow, green, blue, indigo, and violet.',
    hero_image_url: '/images/lessons/rainbow.png'
  },
  12: {
    id: 'emergency-12',
    day_number: 12,
    topic: 'Volcanoes',
    marketing_headline: 'Earth\'s pressure valves',
    marketing_tagline: 'Fire from the deep',
    universal_truth: 'Volcanoes remind us that Earth is still alive and changing.',
    greeting: 'Get ready for an explosive lesson!',
    script: 'Deep underground, rock is so hot it melts into magma. When pressure builds up, it erupts through volcanoes. These eruptions create new land and have shaped our planet for billions of years.',
    hero_image_url: '/images/lessons/volcano.png'
  },
  13: {
    id: 'emergency-13',
    day_number: 13,
    topic: 'Dinosaurs',
    marketing_headline: 'Rulers of the ancient world',
    marketing_tagline: '165 million years of dominance',
    universal_truth: 'Dinosaurs ruled Earth for 165 million years—humans, only 300,000.',
    greeting: 'Let\'s travel back in time 66 million years!',
    script: 'Dinosaurs lived on Earth for 165 million years, from tiny feathered raptors to giant sauropods longer than three school buses. Birds are actually living dinosaurs—the only ones to survive the asteroid.',
    hero_image_url: '/images/lessons/dinosaurs.png'
  },
  14: {
    id: 'emergency-14',
    day_number: 14,
    topic: 'The Ocean',
    marketing_headline: 'Earth\'s blue heart',
    marketing_tagline: 'Covering 71% of our planet',
    universal_truth: 'The ocean holds 97% of all water on Earth.',
    greeting: 'Dive deep with me into the ocean\'s mysteries!',
    script: 'The ocean covers 71% of Earth and holds 97% of all water. Its deepest point, the Mariana Trench, is so deep that Mount Everest could fit inside with room to spare.',
    hero_image_url: '/images/lessons/ocean.png'
  },
  
  // Week 4: Technology & Ideas
  15: {
    id: 'emergency-15',
    day_number: 15,
    topic: 'How Computers Think',
    marketing_headline: 'Billions of tiny switches',
    marketing_tagline: 'Binary brilliance',
    universal_truth: 'Everything a computer does comes down to 1s and 0s.',
    greeting: 'Let\'s peek inside the digital brain!',
    script: 'Computers use transistors—tiny switches that can be on or off, representing 1 or 0. Your phone has billions of these switches, making trillions of calculations every second.',
    hero_image_url: '/images/lessons/computer.png'
  },
  16: {
    id: 'emergency-16',
    day_number: 16,
    topic: 'The Internet',
    marketing_headline: 'The world\'s nervous system',
    marketing_tagline: 'Connecting 5 billion people',
    universal_truth: 'The internet is just computers talking to each other.',
    greeting: 'Let\'s explore the network that connects the world!',
    script: 'The internet is a global network of computers connected by cables, fiber optics, and satellites. When you send a message, it breaks into tiny packets that travel different routes and reassemble at the destination.',
    hero_image_url: '/images/lessons/internet.png'
  },
  17: {
    id: 'emergency-17',
    day_number: 17,
    topic: 'Electricity',
    marketing_headline: 'The invisible force',
    marketing_tagline: 'Powering modern life',
    universal_truth: 'Electricity is the flow of tiny particles called electrons.',
    greeting: 'Spark your curiosity about electricity!',
    script: 'Electricity is the flow of electrons through a conductor. Lightning is natural electricity—a giant spark that can be five times hotter than the surface of the Sun.',
    hero_image_url: '/images/lessons/electricity.png'
  },
  18: {
    id: 'emergency-18',
    day_number: 18,
    topic: 'Gravity',
    marketing_headline: 'The force that shapes the universe',
    marketing_tagline: 'What goes up must come down',
    universal_truth: 'Gravity is what gives weight to everything and keeps planets in orbit.',
    greeting: 'Let\'s explore the force that keeps your feet on the ground!',
    script: 'Gravity is the attraction between all objects with mass. The more massive an object, the stronger its pull. Earth\'s gravity keeps us grounded and the Moon in orbit around us.',
    hero_image_url: '/images/lessons/gravity.png'
  },
  
  // Week 5: Life Skills
  19: {
    id: 'emergency-19',
    day_number: 19,
    topic: 'Why We Laugh',
    marketing_headline: 'The science of joy',
    marketing_tagline: 'Laughter is medicine',
    universal_truth: 'Laughter reduces stress and brings people together.',
    greeting: 'Get ready to smile!',
    script: 'Laughter releases endorphins, reduces stress hormones, and strengthens your immune system. It\'s also contagious—hearing someone laugh makes you 30 times more likely to laugh yourself.',
    hero_image_url: '/images/lessons/laugh.png'
  },
  20: {
    id: 'emergency-20',
    day_number: 20,
    topic: 'How Memory Works',
    marketing_headline: 'Your brain\'s filing system',
    marketing_tagline: 'Remember this!',
    universal_truth: 'Memories aren\'t stored in one place—they\'re patterns across your brain.',
    greeting: 'Let\'s explore how you remember things!',
    script: 'When you experience something, your brain creates connections between neurons. The more you revisit a memory, the stronger these connections become. That\'s why practice makes perfect.',
    hero_image_url: '/images/lessons/memory.png'
  },
  21: {
    id: 'emergency-21',
    day_number: 21,
    topic: 'The Stars',
    marketing_headline: 'Distant suns',
    marketing_tagline: 'Light from billions of years ago',
    universal_truth: 'Every star you see is a sun, some with their own planets.',
    greeting: 'Tonight, look up at the stars and wonder!',
    script: 'The stars you see are other suns, many bigger and brighter than our own. The light from the nearest star takes 4 years to reach us. Some stars you see tonight no longer exist—their light is still traveling.',
    hero_image_url: '/images/lessons/stars.png'
  },
  
  // Extended days (22-30)
  22: { id: 'emergency-22', day_number: 22, topic: 'Sound Waves', universal_truth: 'Sound is vibration traveling through matter.', greeting: 'Listen closely!', script: 'Sound travels as waves of vibration through air, water, or solid materials. That\'s why you can\'t hear anything in space—there\'s nothing to carry the vibrations.' },
  23: { id: 'emergency-23', day_number: 23, topic: 'Photosynthesis', universal_truth: 'Plants turn sunlight into food—and oxygen for us.', greeting: 'Plants are amazing chemists!', script: 'Plants capture sunlight energy and use it to turn water and carbon dioxide into sugar and oxygen. This process feeds almost all life on Earth.' },
  24: { id: 'emergency-24', day_number: 24, topic: 'DNA', universal_truth: 'Your DNA is the instruction manual for building you.', greeting: 'Let\'s decode life itself!', script: 'DNA is a molecule shaped like a twisted ladder. Its rungs spell out instructions in a four-letter code that tells your cells how to build and run your body.' },
  25: { id: 'emergency-25', day_number: 25, topic: 'Ecosystems', universal_truth: 'Every living thing is connected to every other.', greeting: 'Everything is connected!', script: 'An ecosystem is a community where plants, animals, and their environment all depend on each other. Remove one piece and the whole system can change.' },
  26: { id: 'emergency-26', day_number: 26, topic: 'Climate vs Weather', universal_truth: 'Weather is what happens today; climate is the pattern over decades.', greeting: 'What\'s the difference?', script: 'Weather changes day to day—rain, sun, wind. Climate is the average weather over 30+ years. A cold day doesn\'t change climate, just like one warm day doesn\'t mean summer.' },
  27: { id: 'emergency-27', day_number: 27, topic: 'Atoms', universal_truth: 'Everything you see is made of tiny atoms.', greeting: 'Let\'s go incredibly small!', script: 'Atoms are the building blocks of matter. They\'re so small that a single drop of water contains more atoms than there are stars in the Milky Way.' },
  28: { id: 'emergency-28', day_number: 28, topic: 'Time Zones', universal_truth: 'It\'s always daytime somewhere on Earth.', greeting: 'What time is it around the world?', script: 'Earth is divided into 24 time zones because it rotates once every 24 hours. When it\'s noon where you are, it\'s midnight on the opposite side of the planet.' },
  29: { id: 'emergency-29', day_number: 29, topic: 'Kindness', universal_truth: 'Acts of kindness benefit both the giver and receiver.', greeting: 'Kindness is a superpower!', script: 'Being kind releases oxytocin in your brain, making you feel good. Studies show that kind people live longer, have better health, and stronger relationships.' },
  30: { id: 'emergency-30', day_number: 30, topic: 'Curiosity', universal_truth: 'Curiosity is the engine of all learning.', greeting: 'Stay curious forever!', script: 'Curious people learn more, remember better, and find more joy in life. Every great discovery started with someone asking "why?" or "what if?"' }
};

// Merge with basic emergency lessons
if (window.EMERGENCY_LESSONS) {
  window.EMERGENCY_LESSONS = { ...window.EMERGENCY_LESSONS_EXTENDED, ...window.EMERGENCY_LESSONS };
} else {
  window.EMERGENCY_LESSONS = window.EMERGENCY_LESSONS_EXTENDED;
}

console.log('🚨 Emergency Lessons Extended loaded (30 days coverage)');


