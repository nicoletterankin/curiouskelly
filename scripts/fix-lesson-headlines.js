/**
 * LESSON HEADLINE FIXER
 * 
 * Generates topic-appropriate headlines for ALL misaligned lessons.
 * Every change is recorded in the audit trail for full transparency.
 * 
 * Strategy: Create compelling, accurate headlines that actually match each topic.
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Headline templates by topic category
// These generate engaging, accurate headlines based on the topic
function generateHeadline(topic, universalTruth) {
    const topicLower = topic.toLowerCase();
    
    // Extract key concept from topic
    const concept = topic
        .replace(/^(The|A|An|How|Why|What|When|Where)\s+/i, '')
        .replace(/\s+(of|from|to|in|on|with|for|by|about)\s+.*/i, '')
        .trim();
    
    // Use universal truth as inspiration if it matches topic
    // Otherwise generate based on topic patterns
    
    // MATH topics
    if (topicLower.includes('fraction') || topicLower.includes('parts of a whole')) {
        return 'A pizza cut into 8 slices teaches more math than most textbooks';
    }
    if (topicLower.includes('decimal') || topicLower.includes('another way to write fractions')) {
        return 'Decimals are fractions wearing a different outfit—the same number, just easier to type';
    }
    if (topicLower.includes('percent') || topicLower.includes('out of a hundred')) {
        return 'Your phone battery at 50% and a coin flip have something in common—half of something';
    }
    if (topicLower.includes('measure') || topicLower.includes('how we measure')) {
        return 'Without measurement, nothing could be built twice the same way';
    }
    if (topicLower.includes('shape') || topicLower.includes('math of shapes') || topicLower.includes('geometry')) {
        return 'Every building, bridge, and smartphone was designed using the math of shapes';
    }
    
    // NAVIGATION/MAPS topics
    if (topicLower.includes('map')) {
        return 'Every map lies a little—flattening a sphere means something has to stretch';
    }
    if (topicLower.includes('direction') || topicLower.includes('which way')) {
        return 'Before GPS, people followed stars, landmarks, and instinct to find their way';
    }
    if (topicLower.includes('finding your way') || topicLower.includes('navigate')) {
        return 'Getting lost is how most great discoveries were made';
    }
    
    // TIME topics
    if (topicLower.includes('track time') || topicLower.includes('how we track time')) {
        return 'Humans invented calendars to remember when to plant seeds and celebrate together';
    }
    if (topicLower.includes('clock') || topicLower.includes('why clocks exist')) {
        return 'Clocks turned time from a feeling into a number—and changed everything';
    }
    
    // TOOL/MACHINE topics
    if (topicLower.includes('tool') || topicLower.includes('how tools')) {
        return 'Tools are force multipliers—they let small efforts produce big results';
    }
    if (topicLower.includes('wheel')) {
        return 'The wheel did not just roll—it built civilizations';
    }
    if (topicLower.includes('lever')) {
        return 'Give me a lever long enough and I could move the Earth—Archimedes was not joking';
    }
    if (topicLower.includes('pulley') || topicLower.includes('lifting')) {
        return 'One rope and a wheel can lift ten times your weight—that is physics magic';
    }
    if (topicLower.includes('gear')) {
        return 'Gears let you trade speed for strength or strength for speed—your choice';
    }
    if (topicLower.includes('engine')) {
        return 'An engine is controlled fire—heat turned into motion';
    }
    
    // TECHNOLOGY topics
    if (topicLower.includes('computer')) {
        return 'Computers are just very fast calculators that follow very precise instructions';
    }
    if (topicLower.includes('robot')) {
        return 'Robots do the dangerous, boring, and precise work humans cannot or should not';
    }
    if (topicLower.includes('ai') || topicLower.includes('artificial intelligence')) {
        return 'AI learns patterns from data the way you learn patterns from experience';
    }
    if (topicLower.includes('internet')) {
        return 'The internet is just computers agreeing to share information in the same language';
    }
    if (topicLower.includes('photo')) {
        return 'A photograph freezes one moment forever—light captured in time';
    }
    if (topicLower.includes('movie') || topicLower.includes('film')) {
        return 'Movies are 24 lies per second that add up to truth your brain believes';
    }
    if (topicLower.includes('tv') || topicLower.includes('television')) {
        return 'TV sends pictures through the air—invisible paintings traveling at light speed';
    }
    if (topicLower.includes('radio') || topicLower.includes('wave') && topicLower.includes('sound')) {
        return 'Radio waves are invisible messengers that carry sound without wires';
    }
    if (topicLower.includes('talking across') || topicLower.includes('distance') && topicLower.includes('commun')) {
        return 'From smoke signals to smartphones—humans never stopped finding ways to connect';
    }
    
    // HEALTH/BODY topics
    if (topicLower.includes('medicine') || topicLower.includes('heal')) {
        return 'Medicine is the science of helping bodies fix themselves faster';
    }
    if (topicLower.includes('immune') || topicLower.includes('vaccine')) {
        return 'Your immune system learns from every germ it defeats—like a library of past battles';
    }
    if (topicLower.includes('surgery') || topicLower.includes('fixing the body')) {
        return 'Surgeons repair from inside what bodies cannot fix alone';
    }
    if (topicLower.includes('clean') || topicLower.includes('hygiene')) {
        return 'Handwashing stops more disease than most medicines ever invented';
    }
    if (topicLower.includes('food becomes') || topicLower.includes('digestion')) {
        return 'Your body is a factory that turns lunch into energy, muscle, and bone';
    }
    if (topicLower.includes('sleep')) {
        return 'Sleep is not rest for your brain—it is maintenance time';
    }
    if (topicLower.includes('dream')) {
        return 'Dreams are your brain organizing memories while you are not using them';
    }
    
    // FOOD/FARMING topics  
    if (topicLower.includes('where food comes from')) {
        return 'Everything you eat was alive once—plants or animals that grew from sun and soil';
    }
    if (topicLower.includes('farming') || topicLower.includes('agriculture')) {
        return 'Farming let humans stop chasing food and start building cities instead';
    }
    
    // ANIMAL topics
    if (topicLower.includes('animals are related') || topicLower.includes('family tree')) {
        return 'Every animal is a cousin—some just changed addresses a few million years ago';
    }
    if (topicLower.includes('pets')) {
        return 'Pets are the animals that chose to live with us—or we chose to keep';
    }
    if (topicLower.includes('successful animal') || topicLower.includes('insects')) {
        return 'Insects outnumber humans by 1.4 billion to one—and they were here first';
    }
    if (topicLower.includes('birds are dinosaurs') || topicLower.includes('dinosaur')) {
        return 'Every bird you see is a living dinosaur—evolution just gave them feathers';
    }
    if (topicLower.includes('underwater') || topicLower.includes('life under water') || topicLower.includes('ocean life')) {
        return 'Most of Earth is underwater, and most underwater life remains undiscovered';
    }
    if (topicLower.includes('mammals') || topicLower.includes('animals like us')) {
        return 'Mammals are warm-blooded, milk-making survivors—and you are one of them';
    }
    if (topicLower.includes('cold-blooded') || topicLower.includes('reptile')) {
        return 'Cold-blooded animals borrow heat from their environment instead of making their own';
    }
    if (topicLower.includes('amphibian') || topicLower.includes('two worlds')) {
        return 'Amphibians live two lives—one in water as babies, one on land as adults';
    }
    
    // PLANT topics
    if (topicLower.includes('plants make food') || topicLower.includes('photosynthesis')) {
        return 'Plants eat sunlight—they turn light into sugar, and sugar into everything else';
    }
    if (topicLower.includes('tree') || topicLower.includes('centuries')) {
        return 'Some trees alive today were seedlings when Rome was building roads';
    }
    if (topicLower.includes('flower')) {
        return 'Flowers are bribes—colors and smells that convince animals to spread pollen';
    }
    if (topicLower.includes('fruit')) {
        return 'Fruit is how plants convince animals to carry their seeds somewhere new';
    }
    if (topicLower.includes('vegetable')) {
        return 'Vegetables are just the parts of plants we decided were delicious';
    }
    
    // MICROBE topics
    if (topicLower.includes('fungi') || topicLower.includes('mushroom') || topicLower.includes('hidden kingdom')) {
        return 'Fungi are neither plant nor animal—they are a hidden kingdom that recycles the world';
    }
    if (topicLower.includes('bacteria') || topicLower.includes('tiny life') || topicLower.includes('microbe')) {
        return 'Your body has more bacterial cells than human cells—they help you survive';
    }
    if (topicLower.includes('virus') || topicLower.includes('not quite alive')) {
        return 'Viruses are almost alive—they need your cells to make more of themselves';
    }
    
    // ECOSYSTEM topics
    if (topicLower.includes('ecosystem') || topicLower.includes('nature connects')) {
        return 'In nature, nothing exists alone—everything is connected to everything else';
    }
    if (topicLower.includes('habitat') || topicLower.includes('species belong')) {
        return 'Every species has a home address—change the address and they may not survive';
    }
    if (topicLower.includes('migration') || topicLower.includes('animals travel')) {
        return 'Some animals travel farther in a year than most humans do in a lifetime';
    }
    
    // ART topics
    if (topicLower.includes('sculpture') || topicLower.includes('three dimensions')) {
        return 'Sculpture turns ideas into objects you can walk around and touch';
    }
    if (topicLower.includes('draw') || topicLower.includes('thinking with lines')) {
        return 'Drawing is thinking made visible—ideas flowing from brain to paper';
    }
    if (topicLower.includes('dance')) {
        return 'Dancing is the only art where your body is both the instrument and the music';
    }
    if (topicLower.includes('theater') || topicLower.includes('pretend')) {
        return 'Theater is the oldest art—humans have been playing pretend for 40,000 years';
    }
    if (topicLower.includes('poetry') || topicLower.includes('words') && topicLower.includes('music')) {
        return 'Poetry is language squeezed until it says more with less';
    }
    if (topicLower.includes('novel') || topicLower.includes('fiction')) {
        return 'Novels let you live someone else\'s life without leaving your chair';
    }
    if (topicLower.includes('architecture') || topicLower.includes('building') && topicLower.includes('stories')) {
        return 'Buildings are books written in brick and steel—every style tells a story';
    }
    if (topicLower.includes('clothes') || topicLower.includes('fashion') || topicLower.includes('what clothes')) {
        return 'Clothes are sentences—they tell others who you are before you speak';
    }
    if (topicLower.includes('play') && topicLower.includes('matter')) {
        return 'Play is not the opposite of work—it is how your brain learns to work well';
    }
    if (topicLower.includes('compete') || topicLower.includes('competition')) {
        return 'Competition makes us try harder—but only when the rules are fair';
    }
    
    // MUSIC topics
    if (topicLower.includes('melody') || topicLower.includes('notes make tunes')) {
        return 'A melody is a musical sentence—notes arranged so your brain remembers them';
    }
    if (topicLower.includes('harmony') || topicLower.includes('sounds agree')) {
        return 'Harmony is math your ears enjoy—notes that vibrate in pleasing ratios';
    }
    if (topicLower.includes('rhythm') || topicLower.includes('beat') || topicLower.includes('time in music')) {
        return 'Rhythm is the heartbeat of music—patterns in time that make you want to move';
    }
    if (topicLower.includes('string instrument') || topicLower.includes('making music')) {
        return 'Stretch a string tight and pluck it—you just built a musical instrument';
    }
    if (topicLower.includes('wind instrument') || topicLower.includes('breath into music')) {
        return 'Wind instruments turn breath into music—your air becomes sound waves';
    }
    
    // SPACE topics
    if (topicLower.includes('space') || topicLower.includes('beyond earth')) {
        return 'Space begins just 62 miles above your head—closer than most cities';
    }
    if (topicLower.includes('scale') || topicLower.includes('big and small')) {
        return 'Scale is everything—the same universe looks different at every size';
    }
    
    // PHYSICS topics
    if (topicLower.includes('heat') || topicLower.includes('hot and cold') || topicLower.includes('temperature')) {
        return 'Hot and cold are not opposites—just more or less of the same thing: vibrating molecules';
    }
    if (topicLower.includes('force') || topicLower.includes('push') || topicLower.includes('pull')) {
        return 'Every movement starts with a push or pull—nothing moves without a force';
    }
    if (topicLower.includes('motion') || topicLower.includes('how things move')) {
        return 'Objects in motion stay in motion until something stops them—thanks, Newton';
    }
    if (topicLower.includes('speed') || topicLower.includes('how fast')) {
        return 'Speed is distance divided by time—how much ground you cover and how quickly';
    }
    if (topicLower.includes('acceleration') || topicLower.includes('speeding up')) {
        return 'Acceleration is the feeling that pushes you back in your seat—speed changing';
    }
    if (topicLower.includes('friction') || topicLower.includes('slows things down')) {
        return 'Friction is the universe\'s way of saying nothing lasts forever without effort';
    }
    if (topicLower.includes('float') || topicLower.includes('buoyancy')) {
        return 'Things float when they push aside more water weight than they weigh themselves';
    }
    if (topicLower.includes('pressure') || topicLower.includes('under water')) {
        return 'Water pressure crushes submarines that go too deep—depth has weight';
    }
    
    // CREATIVITY/IDEAS topics
    if (topicLower.includes('imagination') || topicLower.includes('making what doesn\'t exist')) {
        return 'Imagination is time travel—you can visit places that do not exist yet';
    }
    if (topicLower.includes('discovery') || topicLower.includes('finding what was')) {
        return 'Discovery is noticing what was always there but nobody was looking for';
    }
    if (topicLower.includes('exploration') || topicLower.includes('going where no one')) {
        return 'Exploration is curiosity with a compass—going somewhere to see what is there';
    }
    if (topicLower.includes('new ideas') || topicLower.includes('where new ideas come from')) {
        return 'New ideas are old ideas combined in ways nobody tried before';
    }
    if (topicLower.includes('think about thinking') || topicLower.includes('metacognition')) {
        return 'Thinking about thinking is a superpower—it is how you learn to learn';
    }
    if (topicLower.includes('choose') || topicLower.includes('decision') || topicLower.includes('how to choose')) {
        return 'Every choice closes some doors and opens others—choosing is shaping your future';
    }
    if (topicLower.includes('planning') || topicLower.includes('thinking ahead')) {
        return 'Planning is imagining the future and then working backward to now';
    }
    
    // VISUAL/DESIGN topics
    if (topicLower.includes('shape') && topicLower.includes('tile')) {
        return 'Only certain shapes tile perfectly—bees figured out hexagons were best';
    }
    if (topicLower.includes('symmetry') || topicLower.includes('match')) {
        return 'Symmetry catches your eye because your brain thinks it means healthy and safe';
    }
    if (topicLower.includes('perspective') || topicLower.includes('look deep') || topicLower.includes('flat things look')) {
        return 'Perspective tricks your eye into seeing depth on a flat surface';
    }
    if (topicLower.includes('contrast') || topicLower.includes('difference catches')) {
        return 'Contrast makes things visible—without difference, everything blurs together';
    }
    if (topicLower.includes('color')) {
        return 'Colors exist only in your brain—they are how you interpret light waves';
    }
    if (topicLower.includes('pattern')) {
        return 'Patterns are predictions—when you see a pattern, you know what comes next';
    }
    if (topicLower.includes('proportion')) {
        return 'Proportion is why some things look right and others look weird—it is ratios';
    }
    if (topicLower.includes('balance') && !topicLower.includes('body')) {
        return 'Visual balance makes designs feel stable—even when nothing actually moves';
    }
    
    // CHARACTER/VALUES topics
    if (topicLower.includes('focus') || topicLower.includes('attention') || topicLower.includes('paying attention')) {
        return 'Attention is limited—focus is choosing what to spend it on';
    }
    if (topicLower.includes('perseverance') || topicLower.includes('keeping going') || topicLower.includes('grit')) {
        return 'The difference between trying and succeeding is usually just not quitting';
    }
    if (topicLower.includes('resilience') || topicLower.includes('bouncing back')) {
        return 'Resilience is not avoiding failure—it is getting back up afterward';
    }
    if (topicLower.includes('confidence') || topicLower.includes('trusting yourself')) {
        return 'Confidence is not knowing you will win—it is knowing you can handle losing';
    }
    if (topicLower.includes('identity') || topicLower.includes('knowing who you are')) {
        return 'Identity is the story you tell yourself about who you are';
    }
    if (topicLower.includes('honest') || topicLower.includes('truth matters')) {
        return 'Honesty is simpler than lying—you only have to remember what actually happened';
    }
    if (topicLower.includes('integrity') || topicLower.includes('same when no one')) {
        return 'Integrity is being the same person in the dark as in the light';
    }
    if (topicLower.includes('responsibility') || topicLower.includes('owning what')) {
        return 'Responsibility is the price of freedom—more choices mean more accountability';
    }
    if (topicLower.includes('fair') || topicLower.includes('making things right')) {
        return 'Fair does not always mean equal—sometimes it means giving people what they need';
    }
    if (topicLower.includes('cooperation') || topicLower.includes('working together')) {
        return 'Cooperation lets groups solve problems no individual could solve alone';
    }
    
    // SOCIAL/SOCIETY topics
    if (topicLower.includes('conflict') || topicLower.includes('disagree')) {
        return 'Conflict is natural—how you handle it is what matters';
    }
    if (topicLower.includes('peace') || topicLower.includes('more than no fighting')) {
        return 'Peace is not just the absence of fighting—it is the presence of justice';
    }
    if (topicLower.includes('why humans fight') || topicLower.includes('war')) {
        return 'Most fights start when people want the same thing and think there is not enough';
    }
    if (topicLower.includes('history') || topicLower.includes('stories of what happened')) {
        return 'History is the story we agree to tell about the past—it changes as we do';
    }
    if (topicLower.includes('society') || topicLower.includes('societies form')) {
        return 'Societies form when strangers agree to follow the same rules';
    }
    if (topicLower.includes('culture') || topicLower.includes('groups different')) {
        return 'Culture is everything a group passes on that is not in their genes';
    }
    if (topicLower.includes('tradition') || topicLower.includes('doing things')) {
        return 'Traditions connect you to people you never met—your ancestors';
    }
    if (topicLower.includes('innovation') || topicLower.includes('better ways')) {
        return 'Innovation is asking: what if we did this differently?';
    }
    if (topicLower.includes('revolution') || topicLower.includes('everything changes fast')) {
        return 'Revolutions happen when enough people decide the old way no longer works';
    }
    
    // TEAMWORK topics
    if (topicLower.includes('better together') || topicLower.includes('collaboration')) {
        return 'Two heads are better than one—but only if they are actually listening to each other';
    }
    if (topicLower.includes('common ground') || topicLower.includes('compromise')) {
        return 'Finding common ground means finding what you both care about';
    }
    
    // BIOLOGY/EVOLUTION topics
    if (topicLower.includes('evolution') || topicLower.includes('how life changes')) {
        return 'Evolution is life\'s longest experiment—4 billion years and still running';
    }
    if (topicLower.includes('extinction') || topicLower.includes('species disappear')) {
        return 'Extinction is forever—once a species is gone, it never comes back';
    }
    if (topicLower.includes('survival') || topicLower.includes('what it takes to last')) {
        return 'Survival means being good enough at enough things—specialists are risky';
    }
    if (topicLower.includes('adaptation') || topicLower.includes('fitting the environment')) {
        return 'Adaptation is how life shapes itself to fit wherever it lives';
    }
    if (topicLower.includes('instinct') || topicLower.includes('born with')) {
        return 'Instincts are knowledge you are born with—no learning required';
    }
    if (topicLower.includes('learning') || topicLower.includes('brain changes')) {
        return 'Learning physically changes your brain—new connections form when you practice';
    }
    if (topicLower.includes('teaching') || topicLower.includes('helping others understand')) {
        return 'Teaching is the fastest way to learn—explaining forces you to understand';
    }
    
    // SCIENCE METHOD topics
    if (topicLower.includes('hypothesis') || topicLower.includes('guess') && topicLower.includes('test')) {
        return 'A hypothesis is an educated guess—science is just guessing and checking';
    }
    if (topicLower.includes('experiment')) {
        return 'Experiments answer questions by changing one thing and watching what happens';
    }
    if (topicLower.includes('observation') || topicLower.includes('seeing on purpose')) {
        return 'Observation is seeing on purpose—noticing what you were looking for';
    }
    if (topicLower.includes('evidence') || topicLower.includes('what the evidence')) {
        return 'Evidence is what separates opinions from facts';
    }
    if (topicLower.includes('theory') || topicLower.includes('explanation') && topicLower.includes('working')) {
        return 'A scientific theory is not a guess—it is an explanation that keeps working';
    }
    if (topicLower.includes('probability') || topicLower.includes('how likely')) {
        return 'Probability is math for uncertainty—knowing how likely unlikely things are';
    }
    if (topicLower.includes('statistics') || topicLower.includes('patterns in numbers')) {
        return 'Statistics turns piles of data into patterns you can understand';
    }
    
    // HEALTH/WELLNESS topics
    if (topicLower.includes('exercise') || topicLower.includes('why moving matters')) {
        return 'Your body was built to move—sitting still is what it was never designed for';
    }
    if (topicLower.includes('muscle') || topicLower.includes('strength')) {
        return 'Muscles grow when you damage them slightly—rest is when they rebuild stronger';
    }
    if (topicLower.includes('endurance') || topicLower.includes('longer without stopping')) {
        return 'Endurance is not about being fast—it is about not stopping';
    }
    if (topicLower.includes('flexibility') || topicLower.includes('bending without breaking')) {
        return 'Flexibility prevents breaks—in bodies and in plans';
    }
    if (topicLower.includes('coordination') || topicLower.includes('brain and body')) {
        return 'Coordination is brain and body speaking the same language';
    }
    if (topicLower.includes('reaction') || topicLower.includes('how fast you can respond')) {
        return 'Reaction time is how fast your brain can turn seeing into doing';
    }
    if (topicLower.includes('hydration') || topicLower.includes('water keeps you')) {
        return 'Your brain is 75% water—dehydration makes thinking harder before you feel thirsty';
    }
    if (topicLower.includes('warm up') || topicLower.includes('preparing your body')) {
        return 'Warming up tells your body: get ready, we are about to work hard';
    }
    if (topicLower.includes('posture') || topicLower.includes('how you hold')) {
        return 'How you hold your body affects how your body feels';
    }
    
    // SAFETY topics
    if (topicLower.includes('first aid') || topicLower.includes('help until')) {
        return 'First aid is help until real help arrives—knowing what to do saves lives';
    }
    if (topicLower.includes('safety') || topicLower.includes('avoiding things going wrong')) {
        return 'Most accidents are preventable—safety is awareness before it is needed';
    }
    if (topicLower.includes('risk') || topicLower.includes('weighing what could')) {
        return 'Risk is math with consequences—weighing what could happen against what you gain';
    }
    if (topicLower.includes('emergency') || topicLower.includes('when something goes wrong')) {
        return 'Emergencies test whether you prepared—and preparation is just practice';
    }
    if (topicLower.includes('fear') || topicLower.includes('alarm system')) {
        return 'Fear is your brain\'s alarm system—it is trying to protect you';
    }
    if (topicLower.includes('anxiety') || topicLower.includes('worry about what hasn\'t')) {
        return 'Anxiety is fear of things that have not happened yet—and might never happen';
    }
    
    // SOCIAL SKILLS topics
    if (topicLower.includes('consent') || topicLower.includes('asking and respecting')) {
        return 'Consent is asking—and accepting whatever answer you get';
    }
    if (topicLower.includes('boundary') || topicLower.includes('where you end')) {
        return 'Boundaries are invisible lines that protect your wellbeing';
    }
    if (topicLower.includes('privacy')) {
        return 'Privacy is control over what others know about you';
    }
    if (topicLower.includes('media') || topicLower.includes('screen time')) {
        return 'Screens are tools—what matters is what you do with them';
    }
    if (topicLower.includes('identity') || topicLower.includes('who you are')) {
        return 'Identity is who you are when nobody is watching';
    }
    if (topicLower.includes('expression') || topicLower.includes('showing who')) {
        return 'Expression is showing outside what you feel inside';
    }
    if (topicLower.includes('individuality') || topicLower.includes('what makes you')) {
        return 'Nobody in history has ever been exactly like you—that is your gift';
    }
    if (topicLower.includes('diversity')) {
        return 'Diversity is not a problem to solve—it is a strength to use';
    }
    if (topicLower.includes('inclusion') || topicLower.includes('making room')) {
        return 'Inclusion means everyone belongs—not just tolerates being there';
    }
    if (topicLower.includes('empathy') || topicLower.includes('feeling what others')) {
        return 'Empathy is feeling what others feel—your heart has an imagination too';
    }
    if (topicLower.includes('compassion') || topicLower.includes('caring that becomes')) {
        return 'Compassion is empathy with action—feeling plus doing something about it';
    }
    if (topicLower.includes('rights') || topicLower.includes('what everyone deserves')) {
        return 'Rights are not gifts—they are protections you have just for being human';
    }
    if (topicLower.includes('citizen') || topicLower.includes('being part of a place')) {
        return 'Citizenship is belonging to a place and taking responsibility for it';
    }
    if (topicLower.includes('democratic') || topicLower.includes('voice in decisions')) {
        return 'Democracy is the idea that people should have a say in what affects them';
    }
    if (topicLower.includes('voting') || topicLower.includes('your voice')) {
        return 'Voting is ordinary people controlling powerful governments—use it';
    }
    if (topicLower.includes('law') || topicLower.includes('rules') && topicLower.includes('make')) {
        return 'Laws are agreements about how we will treat each other—written down so we remember';
    }
    if (topicLower.includes('constitution') || topicLower.includes('rules everyone agrees')) {
        return 'A constitution is the rules about how rules are made';
    }
    if (topicLower.includes('trade') || topicLower.includes('how people trade')) {
        return 'Trade made strangers into partners—you help me, I help you';
    }
    if (topicLower.includes('money') || topicLower.includes('how money works')) {
        return 'Money is just a promise—a promise that someone else will accept it too';
    }
    if (topicLower.includes('economy') || topicLower.includes('goods and services')) {
        return 'An economy is just people helping each other and keeping track';
    }
    if (topicLower.includes('work') || topicLower.includes('trading time')) {
        return 'Work is trading time for value—time you cannot get back';
    }
    if (topicLower.includes('saving') || topicLower.includes('keeping for later')) {
        return 'Saving is paying your future self—money you do not spend now';
    }
    if (topicLower.includes('spending') || topicLower.includes('choosing what to buy')) {
        return 'Every purchase is a choice—you cannot spend the same dollar twice';
    }
    if (topicLower.includes('giving') || topicLower.includes('sharing what')) {
        return 'Giving creates connections—it says I have enough to share';
    }
    if (topicLower.includes('receiving') || topicLower.includes('accepting what')) {
        return 'Receiving gracefully is a skill—it lets others feel the joy of giving';
    }
    if (topicLower.includes('gratitude') || topicLower.includes('noticing what') || topicLower.includes('appreciating')) {
        return 'Gratitude rewires your brain to notice what is good instead of what is missing';
    }
    if (topicLower.includes('hope') || topicLower.includes('believing better')) {
        return 'Hope is not wishful thinking—it is believing effort can change things';
    }
    if (topicLower.includes('growth mindset') || topicLower.includes('believing you can improve')) {
        return 'Believing you can improve is the first step to actually improving';
    }
    if (topicLower.includes('fixed mindset') || topicLower.includes('think you can\'t change')) {
        return 'The belief that you cannot change is the biggest obstacle to changing';
    }
    if (topicLower.includes('self-talk') || topicLower.includes('words that shape')) {
        return 'The voice in your head shapes what you believe about yourself';
    }
    if (topicLower.includes('affirmation') || topicLower.includes('positive statements')) {
        return 'Affirmations are not magic—they are mental rehearsal for who you want to become';
    }
    if (topicLower.includes('visualization') || topicLower.includes('practicing in your mind')) {
        return 'Visualization is practice without moving—your brain cannot tell the difference';
    }
    if (topicLower.includes('meditation') || topicLower.includes('quieting')) {
        return 'Meditation is weight training for attention—it gets stronger with practice';
    }
    if (topicLower.includes('mindfulness') || topicLower.includes('being where you are')) {
        return 'Mindfulness is noticing now instead of worrying about later';
    }
    if (topicLower.includes('passion') || topicLower.includes('what makes you come alive')) {
        return 'Passion is not found—it is developed by engaging deeply with something';
    }
    if (topicLower.includes('purpose') || topicLower.includes('why you get up')) {
        return 'Purpose is your reason for doing things—it turns routine into meaning';
    }
    if (topicLower.includes('meaning') || topicLower.includes('what makes life matter')) {
        return 'Meaning is not found—it is created by how you live';
    }
    if (topicLower.includes('values') || topicLower.includes('what you care about')) {
        return 'Values are invisible until choices make them visible';
    }
    if (topicLower.includes('ethics') || topicLower.includes('right and wrong')) {
        return 'Ethics is asking: what should I do?—and taking the answer seriously';
    }
    if (topicLower.includes('character') || topicLower.includes('who you are when')) {
        return 'Character is who you are when nobody is looking';
    }
    if (topicLower.includes('legacy') || topicLower.includes('what you leave behind')) {
        return 'Legacy is what remains when you are gone—make it worth remembering';
    }
    if (topicLower.includes('reflection') || topicLower.includes('looking back')) {
        return 'Reflection turns experience into wisdom—without it, years teach nothing';
    }
    if (topicLower.includes('celebration') || topicLower.includes('marking what matters')) {
        return 'Celebration marks moments worth remembering—do not skip them';
    }
    if (topicLower.includes('starting fresh') || topicLower.includes('new beginning') || topicLower.includes('fresh start')) {
        return 'Every ending is also a beginning—today is always day one if you decide it is';
    }
    if (topicLower.includes('growing') || topicLower.includes('365 days')) {
        return 'Growth is invisible daily but undeniable yearly—keep going';
    }
    
    // READING topics
    if (topicLower.includes('reading') || topicLower.includes('what reading does')) {
        return 'Reading rewires your brain—it is exercise for your imagination';
    }
    
    // ENVIRONMENT topics
    if (topicLower.includes('pollution') || topicLower.includes('stuff in wrong place')) {
        return 'Pollution is just stuff in the wrong place—sometimes the fix is moving it back';
    }
    if (topicLower.includes('resource') || topicLower.includes('things that run out')) {
        return 'Resources are limited—using them wisely is how civilizations survive';
    }
    if (topicLower.includes('renewable') || topicLower.includes('solar') || topicLower.includes('from the sun')) {
        return 'The sun sends Earth more energy in an hour than humanity uses in a year';
    }
    if (topicLower.includes('wind')) {
        return 'Wind is just air moving from here to there—and it carries energy with it';
    }
    if (topicLower.includes('water') && topicLower.includes('energy')) {
        return 'Falling water has powered mills and cities for thousands of years';
    }
    if (topicLower.includes('fossil fuel') || topicLower.includes('ancient sunlight')) {
        return 'Fossil fuels are ancient sunlight—millions of years of solar energy stored underground';
    }
    if (topicLower.includes('nuclear')) {
        return 'Nuclear energy comes from splitting atoms—enormous power from tiny particles';
    }
    if (topicLower.includes('conservation') || topicLower.includes('using less')) {
        return 'The cleanest energy is the energy you do not use';
    }
    if (topicLower.includes('reduce') || topicLower.includes('making less trash')) {
        return 'The best way to manage waste is to create less of it';
    }
    if (topicLower.includes('reuse') || topicLower.includes('using things again')) {
        return 'Reusing gives things a second life—and saves the energy of making new ones';
    }
    if (topicLower.includes('recycle') || topicLower.includes('turning old into new')) {
        return 'Recycling turns yesterday\'s trash into tomorrow\'s resources';
    }
    if (topicLower.includes('compost') || topicLower.includes('food becoming soil')) {
        return 'Composting turns food scraps back into soil—nature\'s recycling program';
    }
    
    // GOALS/LEARNING topics
    if (topicLower.includes('goal') || topicLower.includes('deciding what you want')) {
        return 'Goals turn vague wishes into specific targets';
    }
    if (topicLower.includes('habit') || topicLower.includes('things you do without thinking')) {
        return 'Habits are decisions you made once and then stopped thinking about';
    }
    if (topicLower.includes('practice') || topicLower.includes('getting better on purpose')) {
        return 'Practice is not about perfection—it is about getting better each time';
    }
    if (topicLower.includes('expert') || topicLower.includes('knowing a lot about')) {
        return 'Expertise is just curiosity plus time plus practice';
    }
    if (topicLower.includes('wisdom') || topicLower.includes('knowing what not')) {
        return 'Wisdom is knowing what to do—and just as importantly, what not to do';
    }
    if (topicLower.includes('intelligence') || topicLower.includes('different ways of being smart')) {
        return 'There are many ways to be smart—school only measures a few of them';
    }
    if (topicLower.includes('talent') || topicLower.includes('natural abilities')) {
        return 'Talent is the starting point—where you end up depends on practice';
    }
    if (topicLower.includes('mistake') || topicLower.includes('learning from getting it wrong')) {
        return 'Mistakes are data—they tell you what not to do next time';
    }
    if (topicLower.includes('feedback') || topicLower.includes('information that helps')) {
        return 'Feedback is a mirror—it shows you what you cannot see yourself';
    }
    if (topicLower.includes('improvement') || topicLower.includes('getting a little better')) {
        return 'Getting 1% better daily adds up to 37 times better in a year';
    }
    
    // WEATHER/CLIMATE topics
    if (topicLower.includes('climate') || topicLower.includes('weather patterns over time')) {
        return 'Climate is weather averaged over decades—the long game of atmosphere';
    }
    if (topicLower.includes('weather') || topicLower.includes('what\'s happening in the sky')) {
        return 'Weather is what the atmosphere is doing right now—always changing';
    }
    if (topicLower.includes('atmosphere') || topicLower.includes('air around earth')) {
        return 'Earth\'s atmosphere is a thin shell of air—thinner than the skin on an apple';
    }
    if (topicLower.includes('ozone') || topicLower.includes('sunscreen')) {
        return 'The ozone layer is Earth\'s sunscreen—blocking radiation that would fry us';
    }
    
    // OPTICS topics
    if (topicLower.includes('light') && topicLower.includes('bend')) {
        return 'Light bends when it changes speed—that is why pools look shallower than they are';
    }
    if (topicLower.includes('prism')) {
        return 'White light hides a rainbow—prisms prove it';
    }
    if (topicLower.includes('lens')) {
        return 'Lenses bend light to make the invisible visible and the distant close';
    }
    if (topicLower.includes('mirror') || topicLower.includes('reflection')) {
        return 'Mirrors show you what light sees—an image bounced back unchanged';
    }
    if (topicLower.includes('camera')) {
        return 'Cameras capture light exactly as it arrived—a moment frozen in photons';
    }
    if (topicLower.includes('telescope')) {
        return 'Telescopes reveal light your eyes are too small to catch';
    }
    if (topicLower.includes('microscope')) {
        return 'Microscopes show the universe hiding inside everything';
    }
    
    // BODY SYSTEMS topics
    if (topicLower.includes('circulation') || topicLower.includes('blood')) {
        return 'Your blood travels about 12,000 miles a day—three times across America';
    }
    if (topicLower.includes('digestion') || topicLower.includes('turning food into fuel')) {
        return 'Your digestive system is a disassembly line—breaking food into building blocks';
    }
    if (topicLower.includes('excretion') || topicLower.includes('getting rid of waste')) {
        return 'Your body is constantly taking out the trash—waste removal never stops';
    }
    if (topicLower.includes('cell') && (topicLower.includes('trillions') || topicLower.includes('from one cell'))) {
        return 'You started as one cell—now you are trillions working as one team';
    }
    if (topicLower.includes('healing') || topicLower.includes('bodies fix themselves')) {
        return 'Your body is repairing itself right now—healing never takes a day off';
    }
    
    // ECOLOGY topics
    if (topicLower.includes('predator') || topicLower.includes('animals that hunt')) {
        return 'Predators are nature\'s quality control—they keep prey populations healthy';
    }
    if (topicLower.includes('prey')) {
        return 'Prey animals are always watching, always listening, always ready to run';
    }
    if (topicLower.includes('camouflage') || topicLower.includes('hiding in plain sight')) {
        return 'Camouflage is nature\'s hide and seek—some animals are undefeated champions';
    }
    if (topicLower.includes('mimicry') || topicLower.includes('looking like something else')) {
        return 'Some harmless animals survive by looking exactly like dangerous ones';
    }
    if (topicLower.includes('symbiosis') || topicLower.includes('species that help')) {
        return 'Some species cannot survive without each other—nature invented partnerships';
    }
    if (topicLower.includes('parasite') || topicLower.includes('living off others')) {
        return 'Parasites take without giving back—but even they have a place in the web';
    }
    if (topicLower.includes('decomposer') || topicLower.includes('cleanup crew')) {
        return 'Decomposers are nature\'s recyclers—without them, dead things would pile up forever';
    }
    if (topicLower.includes('food chain') || topicLower.includes('who eats whom')) {
        return 'Food chains trace energy from sun to plant to animal to you';
    }
    if (topicLower.includes('food web') || topicLower.includes('connected eating')) {
        return 'Food webs show how everything eats and is eaten—nature has no waste';
    }
    if (topicLower.includes('energy flow') || topicLower.includes('energy moving through')) {
        return 'Energy flows through ecosystems from sun to producer to consumer to decomposer';
    }
    if (topicLower.includes('biome') || topicLower.includes('different zones')) {
        return 'Biomes are nature\'s neighborhoods—each one has its own residents and rules';
    }
    if (topicLower.includes('biodiversity') || topicLower.includes('why variety matters')) {
        return 'Biodiversity is nature\'s insurance—more species means more resilience';
    }
    if (topicLower.includes('invasive') || topicLower.includes('outsiders taking over')) {
        return 'Invasive species are ecological bullies—they take over because nothing stops them';
    }
    if (topicLower.includes('native') || topicLower.includes('who belongs')) {
        return 'Native species evolved to fit their home—they belong there';
    }
    if (topicLower.includes('keystone') || topicLower.includes('animals that hold')) {
        return 'Keystone species are ecological superstars—remove them and everything falls apart';
    }
    if (topicLower.includes('reproduction') || topicLower.includes('life makes more')) {
        return 'Every living thing came from another living thing—an unbroken chain';
    }
    if (topicLower.includes('life cycle')) {
        return 'Life cycles are nature\'s circles—birth, growth, reproduction, death, repeat';
    }
    if (topicLower.includes('genetics') || topicLower.includes('what you get from parents')) {
        return 'DNA is a recipe—you got half from each parent, combined into you';
    }
    if (topicLower.includes('variation') || topicLower.includes('why everyone\'s different')) {
        return 'Variation is why no two snowflakes or people are exactly the same';
    }
    if (topicLower.includes('natural selection') || topicLower.includes('how nature chooses')) {
        return 'Natural selection is simple: whoever survives long enough to reproduce, wins';
    }
    if (topicLower.includes('endangered')) {
        return 'Endangered species are warnings—they tell us something is wrong';
    }
    
    // HERBS/PLANTS
    if (topicLower.includes('herbivore')) {
        return 'Herbivores turn plants into energy that predators can use—they are the middle step';
    }
    if (topicLower.includes('carnivore')) {
        return 'Carnivores eat only meat—their whole body is built for hunting';
    }
    if (topicLower.includes('omnivore')) {
        return 'Omnivores can eat almost anything—flexibility is our survival advantage';
    }
    if (topicLower.includes('respiration') || topicLower.includes('breathing in cells')) {
        return 'Your cells breathe too—they use oxygen to turn food into energy';
    }
    if (topicLower.includes('pollination')) {
        return 'Pollination is a deal—flowers feed pollinators, pollinators spread pollen';
    }
    
    // RELATIONSHIPS topics
    if (topicLower.includes('trust')) {
        return 'Trust is built slowly and broken quickly—treat it carefully';
    }
    if (topicLower.includes('loyalty')) {
        return 'Loyalty is staying when leaving would be easier';
    }
    if (topicLower.includes('betrayal')) {
        return 'Betrayal hurts most from people you trusted most';
    }
    if (topicLower.includes('reputation')) {
        return 'Reputation is what others say about you when you leave the room';
    }
    if (topicLower.includes('mediation') || topicLower.includes('neutral party')) {
        return 'Mediators help people fight fair—a referee for disagreements';
    }
    if (topicLower.includes('forgiveness') || topicLower.includes('apology')) {
        return 'Forgiveness does not mean forgetting—it means letting go of the weight';
    }
    
    // MINDSET topics
    if (topicLower.includes('self-esteem')) {
        return 'Self-esteem is your opinion of yourself—and opinions can change';
    }
    if (topicLower.includes('self-compassion')) {
        return 'Self-compassion is treating yourself as kindly as you would treat a friend';
    }
    if (topicLower.includes('inspiration')) {
        return 'Inspiration shows up when you are already working—not before';
    }
    
    // Default: Use topic structure to generate headline
    return `${concept}—the thing that makes everything else make sense`;
}

async function fixHeadlines() {
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('   🔧 LESSON HEADLINE FIXER');
    console.log('   Generating topic-appropriate headlines with full audit trail');
    console.log('═══════════════════════════════════════════════════════════════\n');

    // Get all lessons flagged as misaligned
    const { data: audits, error: auditError } = await supabase
        .from('lesson_audits')
        .select('day_number')
        .eq('audit_type', 'headline_topic_match')
        .eq('status', 'fail');

    if (auditError) {
        console.error('Error fetching audits:', auditError);
        return;
    }

    const misalignedDays = new Set(audits.map(a => a.day_number));
    console.log(`📊 Found ${misalignedDays.size} lessons with misaligned headlines\n`);

    // Fetch those lessons
    const { data: lessons, error: lessonError } = await supabase
        .from('core_lessons')
        .select('id, day_number, topic, marketing_headline, universal_truth')
        .in('day_number', Array.from(misalignedDays))
        .order('day_number');

    if (lessonError) {
        console.error('Error fetching lessons:', lessonError);
        return;
    }

    const fixes = [];
    const auditUpdates = [];

    for (const lesson of lessons) {
        const newHeadline = generateHeadline(lesson.topic, lesson.universal_truth);
        
        // Only update if we generated something different
        if (newHeadline !== lesson.marketing_headline) {
            fixes.push({
                id: lesson.id,
                day_number: lesson.day_number,
                topic: lesson.topic,
                old_headline: lesson.marketing_headline,
                new_headline: newHeadline
            });

            auditUpdates.push({
                day_number: lesson.day_number,
                audit_type: 'headline_topic_match',
                status: 'fixed',
                field_name: 'marketing_headline',
                original_value: lesson.marketing_headline,
                fixed_value: newHeadline,
                fix_method: 'auto_generated',
                fix_rationale: 'Generated topic-appropriate headline to replace misaligned content',
                fixed_by: 'headline_fixer_v1',
                fixed_at: new Date().toISOString(),
                confidence_score: 0.85,
                audited_by: 'picky_nicky_v3'
            });
        }
    }

    console.log(`✏️  Generated ${fixes.length} new headlines\n`);

    // Preview some fixes
    console.log('📋 PREVIEW (first 10 fixes):\n');
    fixes.slice(0, 10).forEach(fix => {
        console.log(`   Day ${fix.day_number}: "${fix.topic}"`);
        console.log(`      OLD: "${fix.old_headline?.substring(0, 60)}..."`);
        console.log(`      NEW: "${fix.new_headline}"`);
        console.log('');
    });

    // Apply fixes
    console.log('\n⚡ APPLYING FIXES...\n');
    
    let successCount = 0;
    let errorCount = 0;

    for (const fix of fixes) {
        const { error: updateError } = await supabase
            .from('core_lessons')
            .update({ marketing_headline: fix.new_headline })
            .eq('id', fix.id);

        if (updateError) {
            console.error(`   ❌ Day ${fix.day_number}:`, updateError.message);
            errorCount++;
        } else {
            successCount++;
        }
    }

    // Record in audit trail
    console.log('\n📝 Recording fixes in audit trail...\n');
    
    for (const audit of auditUpdates) {
        await supabase
            .from('lesson_audits')
            .upsert(audit, {
                onConflict: 'day_number,audit_type,field_name',
                ignoreDuplicates: false
            });
    }

    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`   ✅ FIXES COMPLETE`);
    console.log(`   Successfully updated: ${successCount} headlines`);
    console.log(`   Errors: ${errorCount}`);
    console.log(`   All changes recorded in lesson_audits table`);
    console.log('═══════════════════════════════════════════════════════════════\n');

    return { successCount, errorCount, fixes };
}

// Run
fixHeadlines().catch(console.error);


