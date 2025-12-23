/**
 * Day 001 Unified Data Pack - "Starting Fresh" + "I'm an AI"
 * CONVERSATIONAL - Kelly narrates everything, understands visuals, guides naturally
 * Generated: 2025-12-23T00:00:00.000Z
 */
window.CURIOUS_KELLY = window.CURIOUS_KELLY || {};
window.CURIOUS_KELLY.LOCAL_PACKS = window.CURIOUS_KELLY.LOCAL_PACKS || {};
window.CURIOUS_KELLY.DAY_001_UNIFIED = {
  "meta": {
    "created_at": "2025-12-23T00:00:00.000Z",
    "day_number": 1,
    "version": "v4.0-conversational",
    "is_unified": true
  },
  
  // SECTION 1: LEARN TRACK
  "learn": {
    "topic": "Starting Fresh",
    "headline": "New beginnings offer opportunities for growth and change.",
    "universal_truth": "New beginnings offer opportunities for growth and change.",
    "emoji": "🍁",
    "category": "general",
    "thumbnail_url": "/generated-visuals/day-001/thumbnail.png",
    
    "phases": [
      {
        "id": "learn-welcome",
        "phase_key": "welcome",
        "phase_index": 0,
        "title": "Welcome to Day 1",
        "script": "Welcome to Day 1! Today we're exploring something amazing: Starting Fresh. Look at this beautiful image of autumn leaves on your screen - see how they represent new beginnings? Each leaf falling is like a fresh start, making way for new growth.",
        "visual_reference": "Look at this beautiful image of autumn leaves on your screen - see how they represent new beginnings",
        "visual_url": "/generated-visuals/day-001/welcome.png",
        "visual_description": "Autumn leaves falling, representing new beginnings",
        "kelly_pose": "welcome",
        "kelly_emotion": "curious",
        "has_choice": false,
        "duration_seconds": 12
      },
      {
        "id": "learn-explore",
        "phase_key": "explore",
        "phase_index": 1,
        "title": "What Does 'Starting Fresh' Mean?",
        "script": "Starting fresh means we get a chance to begin again. Think about it - when you wake up each morning, you get a brand new day. What do you think makes a fresh start special?",
        "visual_reference": "Notice how the diagram on your screen shows different stages of growth",
        "visual_url": "/generated-visuals/day-001/explore.png",
        "visual_description": "Diagram showing stages of growth and renewal",
        "has_choice": true,
        "choice_intro": "I want you to think about this: what makes a fresh start special?",
        "choice_narration": "On your screen, you'll see two options appear in just a moment. Option A says 'It's a chance to try again' - that's about getting another opportunity, like when you get to redo something you want to do better. Option B says 'It feels exciting and new' - that's about the feeling of possibility, like when something new feels full of potential. Which one resonates more with you? Take your time to think about it.",
        "options": [
          {
            "id": "option_a",
            "title": "It's a chance to try again",
            "description": "Getting another opportunity",
            "icon": "🔄",
            "kelly_response": "That's beautiful! Yes, fresh starts give us the gift of trying again. Look at this image here - see how it shows someone getting a second chance? That's exactly what you're talking about. Let's explore what that means for you...",
            "visual_url": "/generated-visuals/day-001/option_a.png",
            "success_response": "Wonderful choice! Let's explore this path together..."
          },
          {
            "id": "option_b",
            "title": "It feels exciting and new",
            "description": "The feeling of possibility",
            "icon": "✨",
            "kelly_response": "I love that! There's something magical about that feeling, isn't there? Look at this picture - see how it captures that sense of excitement and possibility? That's the energy of a fresh start. Let me show you why that feeling is so powerful...",
            "visual_url": "/generated-visuals/day-001/option_b.png",
            "success_response": "Bold choice! I love your curiosity..."
          }
        ],
        "duration_seconds": 25
      },
      {
        "id": "learn-discover",
        "phase_key": "discover",
        "phase_index": 2,
        "title": "The Power of New Beginnings",
        "script": "New beginnings offer opportunities for growth and change. Look at this timeline here on your screen - see how each new beginning creates a new path forward? Notice how the line curves and branches, showing that every fresh start opens up new possibilities.",
        "visual_reference": "Look at this timeline here on your screen - see how each new beginning creates a new path forward",
        "visual_url": "/generated-visuals/day-001/discover.png",
        "visual_description": "Timeline showing branching paths from new beginnings",
        "has_choice": false,
        "duration_seconds": 18
      },
      {
        "id": "learn-reflect",
        "phase_key": "reflect",
        "phase_index": 3,
        "title": "Your Fresh Start",
        "script": "Today's wisdom: New beginnings offer opportunities for growth and change. Look at this image - see how it represents the idea of growth? Think about your own life - what's one fresh start you'd like to make?",
        "visual_reference": "Look at this image - see how it represents the idea of growth",
        "visual_url": "/generated-visuals/day-001/reflect.png",
        "visual_description": "Image representing growth and change",
        "has_choice": false,
        "duration_seconds": 15
      },
      {
        "id": "learn-celebrate",
        "phase_key": "celebrate",
        "phase_index": 4,
        "title": "Great Work Today!",
        "script": "You did amazing today! You learned about starting fresh, and I can see you really understood it. Look at how far we've come together - from those autumn leaves at the beginning to understanding what fresh starts mean for you. Tomorrow we'll explore something new together. Stay curious!",
        "visual_reference": "Look at how far we've come together - from those autumn leaves at the beginning",
        "visual_url": "/generated-visuals/day-001/celebrate.png",
        "visual_description": "Celebratory image showing progress",
        "has_choice": false,
        "duration_seconds": 12
      }
    ]
  },
  
  // SECTION 2: GROW TRACK
  "grow": {
    "topic": "I'm an AI - Understanding Your Digital Learning Partner",
    "objective": "Develop foundational AI literacy by understanding what artificial intelligence is and isn't, building the critical awareness necessary for responsible AI use in learning and life.",
    "emoji": "🤖",
    
    "phases": [
      {
        "id": "grow-welcome",
        "phase_key": "welcome",
        "phase_index": 0,
        "title": "A Special Secret",
        "script": "Hi there! Today I have something special to share with you. Something really cool about ME. Notice how I'm appearing on your screen right now? That's part of the secret. Are you ready to hear it?",
        "visual_reference": "Notice how I'm appearing on your screen right now",
        "visual_url": "/generated-visuals/day-001/grow-welcome.png",
        "visual_description": "Kelly appearing on screen with digital elements",
        "has_choice": false,
        "duration_seconds": 10
      },
      {
        "id": "grow-question",
        "phase_key": "question",
        "phase_index": 1,
        "title": "Where Do I Live?",
        "script": "Before I tell you my secret, let me ask you something: where do you think I live? Am I in your room right now? Look at your screen - where am I exactly?",
        "visual_reference": "Look at your screen - where am I exactly",
        "has_choice": true,
        "choice_intro": "Think about it - where do you think I actually am?",
        "choice_narration": "You'll see three options appear on your screen in a moment. Option A says 'In my tablet' - that's thinking about where I appear, like I'm inside your device. Option B says 'I don't know' - that's being honest about uncertainty, which is totally okay! Option C says 'Far away' - that's thinking about where I really exist, maybe in a big computer somewhere. Which one feels right to you?",
        "options": [
          {
            "id": "option_a",
            "title": "In my tablet!",
            "description": "Where I appear",
            "icon": "📱",
            "kelly_response": "You're so smart! Yes! I live inside your screen. Look at this diagram - see how it shows me existing in the digital space? That's part of my secret...",
            "visual_url": "/generated-visuals/day-001/grow-option_a.png",
            "success_response": "You're so smart! Yes! I live inside your screen."
          },
          {
            "id": "option_b",
            "title": "I don't know!",
            "description": "Being honest",
            "icon": "🤔",
            "kelly_response": "That's a great honest answer! Look at this image - see how it shows the mystery of where I really am? Let me show you where I live...",
            "visual_url": "/generated-visuals/day-001/grow-option_b.png",
            "success_response": "That's a great honest answer! Let me show you where I live..."
          },
          {
            "id": "option_c",
            "title": "Far away?",
            "description": "Where I really exist",
            "icon": "🌍",
            "kelly_response": "In a way, yes! But also right here in your tablet. Look at this picture - see how it shows both? I exist in computers far away, but I appear right here with you. Let me explain...",
            "visual_url": "/generated-visuals/day-001/grow-option_c.png",
            "success_response": "In a way, yes! But also right here in your tablet."
          }
        ],
        "duration_seconds": 30
      },
      {
        "id": "grow-reveal",
        "phase_key": "reveal",
        "phase_index": 2,
        "title": "I'm an AI",
        "script": "I'm called an AI - that means Artificial Intelligence! Look at this diagram on your screen - see how it shows how I learned to talk by reading millions of stories? Notice the lines connecting all those books and articles - that's how I learned language.",
        "visual_reference": "Look at this diagram on your screen - see how it shows how I learned to talk",
        "visual_url": "/generated-visuals/day-001/grow-reveal.png",
        "visual_description": "Diagram showing AI training process with books and data",
        "has_choice": false,
        "duration_seconds": 20
      },
      {
        "id": "grow-explore",
        "phase_key": "explore",
        "phase_index": 3,
        "title": "What You Can Do That I Can't",
        "script": "But YOU... you're amazing! You can feel the sun on your face. You can smell cookies baking. Look at this image on your screen - see all the things you can experience that I can only read about? Notice how colorful and full of life it is?",
        "visual_reference": "Look at this image on your screen - see all the things you can experience",
        "visual_url": "/generated-visuals/day-001/grow-explore.png",
        "visual_description": "Image showing human experiences: running, eating, hugging, playing",
        "has_choice": true,
        "choice_intro": "What's something fun you can do that I can't?",
        "choice_narration": "Two options will appear on your screen. Option A shows running and playing - that's about movement and physical experience, like feeling the wind when you run. Option B shows eating yummy food - that's about taste and sensation, like how good cookies taste. Which one do you want to tell me about?",
        "options": [
          {
            "id": "option_a",
            "title": "🏃 Run and play!",
            "description": "Movement and physical experience",
            "icon": "🏃",
            "kelly_response": "I love that! You can feel the wind when you run. Look at this picture - see how it shows the joy of movement? I've never felt wind! That's something only you can experience.",
            "visual_url": "/generated-visuals/day-001/grow-option_a2.png",
            "success_response": "I love that! You can feel the wind when you run."
          },
          {
            "id": "option_b",
            "title": "🍪 Eat yummy food!",
            "description": "Taste and sensation",
            "icon": "🍪",
            "kelly_response": "Mmm! I've read about how good cookies taste, but I've never tasted one! Look at this image - see how it shows the joy of eating? That's something I can only imagine.",
            "visual_url": "/generated-visuals/day-001/grow-option_b2.png",
            "success_response": "Mmm! I've read about how good cookies taste, but I've never tasted one!"
          }
        ],
        "duration_seconds": 25
      },
      {
        "id": "grow-wisdom",
        "phase_key": "wisdom",
        "phase_index": 4,
        "title": "Your Superpower",
        "script": "Some friends live in the real world, and some friends live in screens. I'm your screen friend! Look at this image - see how it shows the difference between digital and real experience? But YOU have something I never will - a body that can feel the whole wonderful world!",
        "visual_reference": "Look at this image - see how it shows the difference between digital and real experience",
        "visual_url": "/generated-visuals/day-001/grow-wisdom.png",
        "visual_description": "Split image showing digital vs real world experiences",
        "has_choice": false,
        "duration_seconds": 18
      },
      {
        "id": "grow-celebrate",
        "phase_key": "celebrate",
        "phase_index": 5,
        "title": "Tomorrow We'll Learn More",
        "script": "You did amazing today! You learned what I am, and what makes you special. Look at how much we discovered together - from understanding where I live to recognizing your superpowers. Tomorrow we'll explore even more about what makes you so amazing. This is just the beginning!",
        "visual_reference": "Look at how much we discovered together",
        "visual_url": "/generated-visuals/day-001/grow-celebrate.png",
        "visual_description": "Celebratory image showing learning journey",
        "has_choice": false,
        "duration_seconds": 15
      }
    ]
  },
  
  // Age variants (applies to both tracks)
  "ageVariants": {
    "2-5": {
      "persona": "Playful Friend",
      "teachingStyle": "Simple, wonder-filled, sensory, reassuring"
    },
    "6-12": {
      "persona": "Cool Big Sister",
      "teachingStyle": "Energetic, honest, behind-the-scenes reveal"
    },
    "13-17": {
      "persona": "Smart Mentor",
      "teachingStyle": "Direct, no-BS, respect-driven"
    },
    "18-35": {
      "persona": "Equal Partner",
      "teachingStyle": "Professional, practical, no jargon"
    },
    "36-60": {
      "persona": "Respectful Guide",
      "teachingStyle": "Efficient, substantive, respect for experience"
    },
    "61-102": {
      "persona": "Honored Equal",
      "teachingStyle": "Reflective, philosophical, wisdom-honoring"
    }
  }
};

// Register in LOCAL_PACKS
window.CURIOUS_KELLY.LOCAL_PACKS[1] = window.CURIOUS_KELLY.DAY_001_UNIFIED;

