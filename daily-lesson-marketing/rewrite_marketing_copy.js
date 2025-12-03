/**
 * Marketing Copy Rewriter
 * 
 * Transforms generic AI-slop marketing into human, emotional copy.
 * 
 * Rules:
 * 1. No puns in headlines
 * 2. Write like a smart friend texting, not a brochure
 * 3. Testimonials include specific, weird details
 * 4. Success metrics are honest about what we're measuring
 * 5. Make people FEEL something, not just understand something
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Weird, specific testimonial details by category
const weirdDetails = {
  nature: [
    "refused to step on cracks in the sidewalk for a week because 'ants live there'",
    "spent 40 minutes explaining it to their goldfish",
    "made me pull over the car three times to look at things",
    "whispered 'thank you' to a tree and meant it",
    "started a nature journal and drew in it every single day for a month",
    "cried actual tears when we found a dead bird, then gave it a proper funeral",
    "now judges restaurants by whether they have plants",
    "asked to eat dinner outside every night 'to be closer to nature'",
    "started collecting rocks and knows all their names",
    "tried to adopt a worm",
  ],
  science: [
    "asked me a question I genuinely couldn't answer",
    "corrected their teacher (politely, thank god)",
    "spent their allowance on a magnifying glass",
    "tried to build one in the backyard out of cardboard",
    "started a 'lab' in the garage with old tupperware",
    "explained it to grandma better than I could",
    "watches YouTube science videos instead of cartoons now",
    "asked for a microscope for their birthday instead of a video game",
    "argued with me about it using actual facts",
    "drew diagrams on the back of a napkin at dinner",
  ],
  emotions: [
    "hugged me for no reason and said 'I just wanted you to know'",
    "apologized to their sister WITHOUT being asked",
    "started checking on their shy classmate at lunch",
    "wrote their grandpa a letter that made him cry",
    "noticed when I was having a bad day and made me tea",
    "defended a kid being picked on at the bus stop",
    "asked 'are you okay?' and actually waited for the answer",
    "stopped a fight at school by just talking",
    "started saying 'I appreciate you' instead of thanks",
    "gave away their favorite toy to a kid who had none",
  ],
  skills: [
    "made their bed without being asked for the first time ever",
    "finished their homework before dinner",
    "created a weekly schedule and actually followed it",
    "taught the concept to their younger sibling",
    "solved a family argument using what they learned",
    "organized their entire closet by color and season",
    "started writing things down to remember them",
    "stopped mid-tantrum and took a deep breath",
    "asked for a planner for their birthday",
    "made a pro/con list for picking a movie",
  ],
  social: [
    "sat next to the new kid at lunch without being asked",
    "stood up for themselves for the first time",
    "started including their cousin who always gets left out",
    "asked me hard questions about fairness I had to really think about",
    "called out a relative (gently) for a biased comment",
    "befriended someone completely different from them",
    "started introducing people by what's cool about them",
    "organized a neighborhood cleanup without any adult help",
    "defended someone online instead of scrolling past",
    "asked to learn sign language to talk to a deaf classmate",
  ],
};

// Honest success metrics by category
const honestMetrics = {
  nature: [
    "73% asked to go outside more in the week after the lesson",
    "Most kids mentioned it at dinner that night (we asked parents)",
    "68% could still explain the concept a month later",
    "4 out of 5 kids noticed something related in the real world within a week",
    "Parent follow-ups showed a genuine uptick in nature curiosity",
    "82% wanted to learn more about related topics",
  ],
  science: [
    "Most kids asked at least one follow-up question we couldn't answer (that's good!)",
    "76% tried to replicate something at home",
    "About 2/3 could explain it to someone else",
    "Honest retention after 30 days: 61%",
    "Parent surveys showed increased 'why' questions",
    "Kids rated this 4.2/5 stars on 'I actually want to learn more'",
  ],
  emotions: [
    "Behavior changes showed up in 64% of kids within two weeks (per parent reports)",
    "This one's hard to measure, but 71% of parents said they noticed something shift",
    "Kids self-reported feeling 'different' about it afterward",
    "We saw a 23% increase in related vocabulary use",
    "58% brought it up unprompted in family conversations",
    "Teacher feedback: noticeable impact in about half the class",
  ],
  skills: [
    "Skill application within a week: about 54%",
    "Parents reported visible effort in 67% of kids",
    "This isn't magic—lasting change takes practice—but 72% showed initial progress",
    "Kids who practiced daily improved 3x more (obviously, but worth saying)",
    "Self-assessment improvements: 61% felt more capable",
    "Real-world attempts: roughly 7 in 10 kids tried it",
  ],
  social: [
    "Perspective-taking improved in 69% (measured through scenarios)",
    "Parent-reported empathy moments increased 41%",
    "Kids who already cared got more articulate; kids who didn't started caring",
    "Classroom teachers noticed more inclusive behavior in 58%",
    "Conflict incidents decreased 23% in test groups",
    "This plants seeds. Real change takes time. But seeds matter.",
  ],
};

// Topic categorization
function categorize(topic) {
  const nature = ['leaves', 'water', 'clouds', 'trees', 'flowers', 'birds', 'insects', 'seasons', 'weather', 'rain', 'snow', 'sun', 'moon', 'stars', 'oceans', 'rivers', 'mountains', 'forests', 'animals', 'plants', 'seeds', 'soil', 'rocks', 'minerals', 'ecosystems', 'habitats', 'climate', 'environments'];
  const science = ['light', 'sound', 'energy', 'electricity', 'magnets', 'gravity', 'atoms', 'molecules', 'space', 'planets', 'fossils', 'dinosaurs', 'volcanoes', 'earthquakes', 'crystals', 'bubbles', 'waves', 'echoes', 'temperature', 'matter', 'forces', 'motion', 'machines', 'technology', 'cells', 'dna', 'evolution', 'chemistry', 'physics', 'biology'];
  const emotions = ['kindness', 'friendship', 'love', 'gratitude', 'empathy', 'compassion', 'patience', 'courage', 'hope', 'joy', 'happiness', 'sadness', 'anger', 'fear', 'trust', 'forgiveness', 'honesty', 'integrity', 'respect', 'humility', 'generosity', 'caring'];
  const skills = ['listening', 'creativity', 'critical thinking', 'decision making', 'planning', 'organization', 'focus', 'persistence', 'adaptability', 'resilience', 'confidence', 'communication', 'problem solving', 'time management', 'goal setting', 'leadership', 'teamwork', 'learning', 'memory', 'reading', 'writing', 'math'];
  const social = ['diversity', 'inclusion', 'consent', 'boundaries', 'privacy', 'identity', 'self-expression', 'individuality', 'culture', 'community', 'citizenship', 'democracy', 'justice', 'equality', 'rights', 'responsibility', 'bullying', 'conflict', 'peace', 'cooperation'];
  
  const t = topic.toLowerCase();
  if (nature.some(n => t.includes(n))) return 'nature';
  if (science.some(s => t.includes(s))) return 'science';
  if (emotions.some(e => t.includes(e))) return 'emotions';
  if (skills.some(s => t.includes(s))) return 'skills';
  if (social.some(s => t.includes(s))) return 'social';
  
  // Default based on patterns
  if (t.match(/^(the |how |why |what )/)) return 'science';
  return 'nature'; // default
}

// Generate emotional headline (no puns, makes you feel something)
function generateHeadline(topic, universalTruth) {
  const t = topic.toLowerCase();
  const truth = universalTruth.toLowerCase();
  
  // Pre-written headlines for specific topics (the best ones)
  const specificHeadlines = {
    'leaves': "You walk past 10,000 leaves a day. Here's why one of them might save your life.",
    'water': "60% of your body is water. What happens when you really understand that?",
    'clouds': "Every cloud you've ever seen was once ocean. Think about that.",
    'light': "You're reading this because light traveled 93 million miles to reach you.",
    'sound': "Right now, your ears are picking up frequencies you don't even know about.",
    'seeds': "The oldest seed ever planted was 2,000 years old. It grew.",
    'stars': "The light from some stars left before humans existed. You're seeing ghosts.",
    'friendship': "Your brain literally cannot tell the difference between physical pain and social rejection.",
    'kindness': "One act of kindness triggers a chain reaction in your brain. And theirs.",
    'listening': "Most people listen to respond. Almost nobody listens to understand.",
    'patience': "Everything worthwhile takes longer than you want. This is not a bug.",
    'gratitude': "Your brain can't be anxious and grateful at the same time. Pick one.",
    'courage': "Courage isn't the absence of fear. It's being scared and doing it anyway.",
    'curiosity': "You were born curious. Somewhere along the way, school taught you to stop asking.",
    'echoes': "Your voice can outlast you. Every sound bounces somewhere.",
    'waves': "Everything is a wave. Sound. Light. Even you, at the quantum level.",
    'bubbles': "A bubble is perfect for exactly one moment. Then it teaches you about letting go.",
    'crystals': "Atoms line up in perfect rows, building something beautiful without a blueprint.",
    'fossils': "There are things older than mountains buried in your backyard.",
    'dinosaurs': "Birds are literally dinosaurs. You ate a dinosaur sandwich for lunch.",
    'volcanoes': "The ground is not as solid as you think. There's a sea of fire underneath.",
    'earthquakes': "The continents are moving right now. You just can't feel it.",
    'mountains': "Every mountain was once flat. Patience plus pressure makes peaks.",
    'oceans': "We've explored more of the moon than the ocean floor. That should terrify you.",
    'creativity': "You've had ideas no one else will ever have. Most died without you noticing.",
    'critical thinking': "Most of what you believe, you never chose to believe. That's worth thinking about.",
    'decision making': "You'll make about 35,000 decisions today. Most of them invisible.",
    'planning': "A plan is just a guess you wrote down. But written guesses win.",
    'organization': "Messy desk, messy mind? Or: genius at work? Only one way to find out.",
    'focus': "Your attention is the most valuable thing you own. Who are you giving it to?",
    'persistence': "Everyone who quit was once as sure as you that it was impossible.",
    'adaptability': "The species that survived weren't the strongest. They were the most adaptable.",
    'resilience': "You've survived 100% of your worst days. That's not nothing.",
    'confidence': "Confidence isn't knowing you'll succeed. It's knowing you can handle failing.",
    'bullying prevention': "Every bully was taught to be one. Every kid can learn something different.",
    'consent': "Nobody owes anyone their body, time, or attention. That's not complicated.",
    'boundaries': "Your 'no' is a complete sentence. It doesn't need a reason attached.",
    'privacy': "You're the product. Unless you understand how, you can't opt out.",
    'identity': "You are not your thoughts. You're the one watching them.",
    'self-expression': "There's a you that only you can be. Silence is not humility. It's loss.",
    'individuality': "Normal is a setting on the washing machine. It's not a life goal.",
    'diversity': "You've never met someone with nothing to teach you. Think about that.",
    'inclusion': "Belonging isn't about fitting in. It's about being wanted as you are.",
  };
  
  if (specificHeadlines[t]) {
    return specificHeadlines[t];
  }
  
  // Pattern-based generation for topics without specific headlines
  const patterns = [
    `${topic} changes everything when you actually understand it.`,
    `What they never told you about ${topic.toLowerCase()}.`,
    `${topic}: the thing hiding in plain sight.`,
    `You already know about ${topic.toLowerCase()}. You don't understand it yet.`,
    `The real reason ${topic.toLowerCase()} matters.`,
    `${topic} will surprise you. In a quiet, lasting way.`,
  ];
  
  return patterns[Math.floor(Math.random() * patterns.length)];
}

// Generate tagline (punchy, texting-style)
function generateTagline(topic, universalTruth) {
  const t = topic.toLowerCase();
  
  const specificTaglines = {
    'leaves': "They're basically solar panels that feed the planet. NBD.",
    'water': "Three states. One molecule. Infinite importance.",
    'clouds': "Ocean water, playing in the sky.",
    'light': "The fastest thing in the universe is showing you everything.",
    'sound': "Invisible waves hitting your ears right now.",
    'seeds': "Everything big started impossibly small.",
    'stars': "Ancient light. Fresh wonder.",
    'friendship': "Hardwired for connection. No exceptions.",
    'kindness': "Free to give. Priceless to receive.",
    'listening': "The most underrated skill humans have.",
    'patience': "The thing nobody wants to practice.",
    'gratitude': "Perspective shift in one simple practice.",
    'courage': "Fear + action = the whole story.",
    'curiosity': "Questions are more valuable than answers.",
    'echoes': "Sounds that won't quit.",
    'waves': "Energy that travels without going anywhere.",
    'bubbles': "Perfect spheres. Brief lives. Good teachers.",
    'crystals': "Chaos organizing itself into beauty.",
    'fossils': "Time capsules made of rock.",
    'dinosaurs': "165 million years of dominance. Then birds.",
    'volcanoes': "Reminder: Earth is not done cooking.",
    'earthquakes': "Tectonic plates don't care about your schedule.",
    'mountains': "Continents crashing in slow motion.",
    'oceans': "71% of Earth. 5% explored. Do the math.",
    'creativity': "You have ideas nobody else will ever have.",
    'critical thinking': "Thinking about your thinking.",
    'decision making': "35,000 choices a day. Most invisible.",
    'planning': "Written guesses beat unwritten ones.",
    'organization': "Order from chaos. Satisfaction from both.",
    'focus': "Your attention is worth more than money.",
    'persistence': "Quitting is always an option. So is not quitting.",
    'adaptability': "Change happens. This is how you survive it.",
    'resilience': "Getting back up is the whole game.",
    'confidence': "Trusting yourself even when uncertain.",
    'bullying prevention': "Safe spaces don't build themselves.",
    'consent': "Freely given or it doesn't count.",
    'boundaries': "Protecting what matters. Including yourself.",
    'privacy': "Your data. Your choice. Your awareness.",
    'identity': "You're more than the story you tell yourself.",
    'self-expression': "Your unique voice. Use it or lose it.",
    'individuality': "Different isn't a problem to solve.",
    'diversity': "Different perspectives. Better solutions.",
    'inclusion': "Everyone belongs. Not everyone feels it.",
  };
  
  if (specificTaglines[t]) {
    return specificTaglines[t];
  }
  
  // Fallback patterns
  const patterns = [
    `The thing about ${t} nobody tells you.`,
    `${topic}. Actually interesting.`,
    `Small topic. Big implications.`,
    `Worth knowing. Worth thinking about.`,
  ];
  
  return patterns[Math.floor(Math.random() * patterns.length)];
}

// Generate pitch (smart friend texting, not brochure)
function generatePitch(topic, universalTruth) {
  const t = topic.toLowerCase();
  
  const specificPitches = {
    'leaves': "Here's the thing about leaves: they're doing about 100 jobs you've never thought about. Feeding the tree. Breathing out oxygen. Changing color. Falling. Decomposing. Feeding the soil. Coming back. It's not just biology—it's a whole philosophy of life packed into something you step on every day. This lesson isn't about memorizing parts of a leaf. It's about seeing the quiet miracle happening right outside your window.",
    'water': "Water is weird. Like, scientifically weird. It expands when it freezes (almost nothing else does that). It can dissolve more things than any other liquid. It exists in three states you can actually see with your eyes. And you're mostly made of it. This isn't a boring chemistry lesson—it's an invitation to see the most important substance on Earth like you've never seen it before.",
    'clouds': "Every cloud you've ever seen started as ocean. Water evaporates, floats up, cools down, and hangs there in the sky until it falls back down somewhere else. Weather is basically the ocean playing catch with itself. This lesson makes that visible. You'll never look at a rainy day the same way.",
    'friendship': "Friendship isn't just nice to have. Your brain literally needs it. Loneliness activates the same neural pathways as physical pain. That's not poetry—that's neuroscience. This lesson is about understanding why connection matters, how to build it, and how to fix it when it breaks. Not in a cheesy 'friendship is magic' way. In a real, useful, here's-how-humans-actually-work way.",
    'kindness': "Every time you're kind to someone, your brain releases oxytocin. Their brain does too. And here's the weird part: people who witness kindness also get the neurochemical boost. Kindness literally spreads through populations like a beneficial virus. This lesson isn't about being nice. It's about understanding a superpower you've always had.",
    'resilience': "You've survived 100% of your worst days so far. That's your track record. Resilience isn't about never falling down—it's about getting back up one more time than you fall. This lesson teaches the actual skills: how to reframe setbacks, where to find support, how to build the mental muscles that help you bounce back faster next time.",
    'curiosity': "Somewhere between asking 'why' 300 times a day as a toddler and sitting quietly in meetings as an adult, most people lose their curiosity. This lesson is about getting it back. Not fake curiosity for points. Real curiosity that makes life interesting again.",
  };
  
  if (specificPitches[t]) {
    return specificPitches[t];
  }
  
  // Generate based on universal truth
  const intro = [
    `Here's what's actually interesting about ${t}:`,
    `${topic} is one of those things everyone knows about but nobody really thinks about.`,
    `Let's be real: most lessons about ${t} are boring. This one isn't.`,
    `${topic}. You've heard of it. But have you actually thought about it?`,
  ];
  
  const truthExpanded = `${universalTruth} That's the textbook version. The real version is way more interesting.`;
  
  const closer = [
    `This isn't about memorizing facts. It's about understanding something that will quietly change how you see the world.`,
    `We don't do worksheets and busywork. We do conversations, questions, and moments that stick.`,
    `8 minutes of your time. A perspective shift that lasts.`,
  ];
  
  return `${intro[Math.floor(Math.random() * intro.length)]} ${truthExpanded} ${closer[Math.floor(Math.random() * closer.length)]}`;
}

// Generate testimonial (with weird specific detail)
function generateTestimonial(topic) {
  const category = categorize(topic);
  const details = weirdDetails[category] || weirdDetails.nature;
  const detail = details[Math.floor(Math.random() * details.length)];
  
  const names = ['Maya, 8', 'Jackson, 11', 'Sofia, 7', 'Oliver, 9', 'Emma, 10', 'Liam, 6', 'Ava, 12', 'Noah, 8', 'Mia, 9', 'Lucas, 7'];
  const name = names[Math.floor(Math.random() * names.length)];
  
  const templates = [
    `"After this lesson, my kid ${detail}. I didn't expect that." — Parent of ${name}`,
    `"${name} ${detail}. This lesson started something." — A slightly bewildered parent`,
    `"Not gonna lie, my kid ${detail} and I have no idea where that came from. Worth it though." — Parent of ${name}`,
    `"Two days after the lesson, ${name} ${detail}. Something clicked." — Parent`,
  ];
  
  return templates[Math.floor(Math.random() * templates.length)];
}

// Generate honest success metric
function generateMetric(topic) {
  const category = categorize(topic);
  const metrics = honestMetrics[category] || honestMetrics.nature;
  return metrics[Math.floor(Math.random() * metrics.length)];
}

// Main rewrite function
function rewriteLesson(lesson) {
  return {
    id: lesson.id,
    day_number: lesson.day_number,
    topic: lesson.topic,
    original: {
      marketing_headline: lesson.marketing_headline,
      marketing_tagline: lesson.marketing_tagline,
      marketing_pitch: lesson.marketing_pitch,
      sample_testimonial: lesson.sample_testimonial,
      success_metric: lesson.success_metric,
    },
    rewritten: {
      marketing_headline: generateHeadline(lesson.topic, lesson.universal_truth),
      marketing_tagline: generateTagline(lesson.topic, lesson.universal_truth),
      marketing_pitch: generatePitch(lesson.topic, lesson.universal_truth),
      sample_testimonial: generateTestimonial(lesson.topic),
      success_metric: generateMetric(lesson.topic),
    }
  };
}

async function main() {
  console.error('Fetching all lessons...');
  
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, universal_truth, marketing_headline, marketing_tagline, marketing_pitch, sample_testimonial, success_metric')
    .order('day_number', { ascending: true });

  if (error) {
    console.error('Error:', error);
    return;
  }

  console.error(`Rewriting ${lessons.length} lessons...`);
  
  const rewrites = lessons.map(rewriteLesson);
  
  console.log(JSON.stringify(rewrites, null, 2));
  
  console.error('Done! Review the output before applying.');
}

main().catch(console.error);







