/**
 * Day 351 Complete Data Pack - "Practicing in Your Mind"
 * Launch Day: December 17, 2025
 * 
 * FULL STRUCTURE: Each phase includes talk, question, options, responses, comments
 * Adult Kelly only, English only
 * 
 * Audio files needed: 35 (7 phases × 5 audio types)
 */
window.CURIOUS_KELLY = window.CURIOUS_KELLY || {};
window.CURIOUS_KELLY.LOCAL_PACKS = window.CURIOUS_KELLY.LOCAL_PACKS || {};
window.CURIOUS_KELLY.DAY_351 = {
  "meta": {
    "created_at": "2025-12-17T00:00:00.000Z",
    "day_number": 351,
    "version": "v5.0-full-structure",
    "date": "December 17, 2025",
    "language": "en",
    "age_group": "adult",
    "voice_id": "wAdymQH5YucAkXwmrdL0",
    "audio_files_count": 35
  },
  "lesson": {
    "day_number": 351,
    "topic": "Practicing in Your Mind",
    "headline": "Your brain can't tell the difference between doing and imagining",
    "universal_truth": "The mind that rehearses grows stronger than the mind that merely waits",
    "emoji": "🔮",
    "category": "Meta-Learning",
    "total_duration_seconds": 180,
    "kelly_images": {
      "hook": "/kelly/phases/351/hook.png",
      "cliff": "/kelly/phases/351/cliff.png",
      "fact1": "/kelly/phases/351/q1.png",
      "fact2": "/kelly/phases/351/q2.png",
      "fact3": "/kelly/phases/351/q3.png",
      "wisdom": "/kelly/phases/351/wisdom.png",
      "outro": "/kelly/phases/351/outro.png"
    }
  },
  "phases": {
    "hook": {
      "name": "Hook",
      "icon": "🪝",
      "order": 1,
      "talk": {
        "script": "Ever wondered why athletes close their eyes before a big moment? They're not just calming their nerves. They're doing something far more powerful—they're practicing. Without moving a muscle. It's called visualization, and the science behind it might change how you think about learning itself.",
        "duration": 18,
        "audio": "/audio/351/hook_talk.mp3",
        "kellyPose": "curious",
        "kellyEmotion": "intrigued"
      },
      "question": {
        "prompt": "Before we dive in—have you ever imagined doing something before you actually did it?",
        "audio": "/audio/351/hook_question.mp3",
        "duration": 5
      },
      "options": [
        {
          "letter": "A",
          "text": "Yes, I mentally rehearse things sometimes",
          "quality": "best"
        },
        {
          "letter": "B",
          "text": "Not really, I usually just wing it",
          "quality": "good"
        }
      ],
      "responses": {
        "A": {
          "script": "You're already tapping into something powerful. Today you'll learn exactly why that works—and how to do it even better.",
          "audio": "/audio/351/hook_response_a.mp3",
          "duration": 8
        },
        "B": {
          "script": "That's totally normal! Most people don't realize what they're missing. By the end of today, you might change your approach.",
          "audio": "/audio/351/hook_response_b.mp3",
          "duration": 8
        }
      },
      "studentComment": {
        "name": "Jordan",
        "avatar": "👤",
        "text": "I always picture my presentations before giving them. Didn't know there was science behind it!",
        "audio": "/audio/351/hook_comment.mp3",
        "duration": 5
      }
    },
    "cliff": {
      "name": "The Cliff",
      "icon": "🧗",
      "order": 2,
      "talk": {
        "script": "Here's where it gets interesting. When you vividly imagine doing something—really see it, feel it, experience it in your mind—your brain activates almost the same way as when you actually do it. The neurons fire. The pathways light up. But here's the question that puzzled scientists for years...",
        "duration": 20,
        "audio": "/audio/351/cliff_talk.mp3",
        "kellyPose": "explaining",
        "kellyEmotion": "thoughtful"
      },
      "question": {
        "prompt": "Why would imagining something make you better at actually doing it?",
        "audio": "/audio/351/cliff_question.mp3",
        "duration": 4
      },
      "options": [
        {
          "letter": "A",
          "text": "It's probably just confidence—positive thinking",
          "quality": "good"
        },
        {
          "letter": "B",
          "text": "Maybe it actually trains the brain somehow",
          "quality": "best"
        }
      ],
      "responses": {
        "A": {
          "script": "That's what researchers thought at first too! Confidence does play a role. But brain scans revealed something far more concrete happening inside the skull.",
          "audio": "/audio/351/cliff_response_a.mp3",
          "duration": 10
        },
        "B": {
          "script": "Exactly right. And not in some vague, mystical way—we're talking measurable, physical changes in neural structure. Let me show you the evidence.",
          "audio": "/audio/351/cliff_response_b.mp3",
          "duration": 9
        }
      },
      "studentComment": {
        "name": "Maya",
        "avatar": "👤",
        "text": "Wait, so daydreaming might actually be... productive?",
        "audio": "/audio/351/cliff_comment.mp3",
        "duration": 4
      }
    },
    "fact1": {
      "name": "Fact 1: Neural Overlap",
      "icon": "🧠",
      "order": 3,
      "talk": {
        "script": "When you imagine performing an action, your motor cortex—that's the part of your brain that controls movement—lights up almost identically to when you actually move. Brain scans show about 90% overlap. Ninety percent. Your brain literally cannot tell the difference between vividly imagining something and doing it. It's practicing either way.",
        "duration": 22,
        "audio": "/audio/351/fact1_talk.mp3",
        "kellyPose": "explaining",
        "kellyEmotion": "enthusiastic"
      },
      "question": {
        "prompt": "What do you think this means for learning new skills?",
        "audio": "/audio/351/fact1_question.mp3",
        "duration": 4
      },
      "options": [
        {
          "letter": "A",
          "text": "You could practice anywhere, anytime—even without equipment",
          "quality": "best"
        },
        {
          "letter": "B",
          "text": "It might help, but real practice is probably still way better",
          "quality": "good"
        }
      ],
      "responses": {
        "A": {
          "script": "You've got it. On the bus, in bed, waiting in line—your brain doesn't care where your body is. It's ready to train.",
          "audio": "/audio/351/fact1_response_a.mp3",
          "duration": 8
        },
        "B": {
          "script": "Real practice is important, absolutely. But here's the thing—the best performers don't choose one or the other. They combine both. And the results are remarkable.",
          "audio": "/audio/351/fact1_response_b.mp3",
          "duration": 10
        }
      },
      "studentComment": {
        "name": "Alex",
        "avatar": "👤",
        "text": "90%?! That's insane. My brain's been lying to me this whole time.",
        "audio": "/audio/351/fact1_comment.mp3",
        "duration": 5
      }
    },
    "fact2": {
      "name": "Fact 2: The Piano Study",
      "icon": "🎹",
      "order": 4,
      "talk": {
        "script": "Let me tell you about a famous experiment. Researchers took people who had never played piano and divided them into three groups. Group one physically practiced a simple piece for five days. Group two only imagined practicing—same piece, same time, but never touched a key. Group three did nothing. After five days, they scanned everyone's brains. The results shocked the scientific community.",
        "duration": 25,
        "audio": "/audio/351/fact2_talk.mp3",
        "kellyPose": "storytelling",
        "kellyEmotion": "engaged"
      },
      "question": {
        "prompt": "What do you think they found when comparing the imagination group to the physical practice group?",
        "audio": "/audio/351/fact2_question.mp3",
        "duration": 5
      },
      "options": [
        {
          "letter": "A",
          "text": "The imagination group showed some improvement, but way less",
          "quality": "good"
        },
        {
          "letter": "B",
          "text": "Their brains changed almost identically",
          "quality": "best"
        }
      ],
      "responses": {
        "A": {
          "script": "That's the logical guess. But here's the twist—the imagination group's brains showed nearly identical changes to the physical practice group. Mental rehearsal created real, measurable neuroplastic changes.",
          "audio": "/audio/351/fact2_response_a.mp3",
          "duration": 12
        },
        "B": {
          "script": "Exactly. The brain regions responsible for piano playing grew in both groups. Imagination alone rewired their brains. Not as much as physical practice, but remarkably close.",
          "audio": "/audio/351/fact2_response_b.mp3",
          "duration": 11
        }
      },
      "studentComment": {
        "name": "Sam",
        "avatar": "👤",
        "text": "So I can tell my parents I'm practicing piano in my head? 😄",
        "audio": "/audio/351/fact2_comment.mp3",
        "duration": 4
      }
    },
    "fact3": {
      "name": "Fact 3: Elite Practice",
      "icon": "🏆",
      "order": 5,
      "talk": {
        "script": "This isn't just lab science. Elite performers have known this for decades. Olympic athletes spend up to 50% of their training time on mental rehearsal. Surgeons visualize entire procedures before making a single cut. Concert pianists play through pieces in their minds on the flight to performances. The key they all discovered: specificity. Vague daydreaming doesn't work. You need vivid, detailed, multi-sensory imagination.",
        "duration": 28,
        "audio": "/audio/351/fact3_talk.mp3",
        "kellyPose": "passionate",
        "kellyEmotion": "inspired"
      },
      "question": {
        "prompt": "What makes visualization most effective, based on what the pros do?",
        "audio": "/audio/351/fact3_question.mp3",
        "duration": 4
      },
      "options": [
        {
          "letter": "A",
          "text": "Imagining success and positive outcomes",
          "quality": "good"
        },
        {
          "letter": "B",
          "text": "Vivid detail—seeing, feeling, hearing every step",
          "quality": "best"
        }
      ],
      "responses": {
        "A": {
          "script": "Positive outcomes matter for motivation, but here's the secret the pros know: you have to visualize the process, not just the result. Feel the movements. See the environment. Hear the sounds. That's what triggers the neural overlap.",
          "audio": "/audio/351/fact3_response_a.mp3",
          "duration": 14
        },
        "B": {
          "script": "That's the key. The more senses you engage, the more your brain treats it as real practice. See it, feel it, hear it. First-person perspective. Every detail matters.",
          "audio": "/audio/351/fact3_response_b.mp3",
          "duration": 11
        }
      },
      "studentComment": {
        "name": "Riley",
        "avatar": "👤",
        "text": "50% of Olympic training is just... thinking? Mind = blown.",
        "audio": "/audio/351/fact3_comment.mp3",
        "duration": 5
      }
    },
    "wisdom": {
      "name": "Wisdom",
      "icon": "🦉",
      "order": 6,
      "talk": {
        "script": "Here's today's wisdom: Your imagination is a practice field. The mind that rehearses builds pathways the passive mind never develops. Every time you vividly imagine doing something, you're laying down the neural tracks that make it easier to do for real. This is one of the few truly free performance enhancers available to every human being.",
        "duration": 22,
        "audio": "/audio/351/wisdom_talk.mp3",
        "kellyPose": "warm",
        "kellyEmotion": "wise"
      },
      "question": {
        "prompt": "What's one skill you'd like to practice in your mind this week?",
        "audio": "/audio/351/wisdom_question.mp3",
        "duration": 4
      },
      "options": [
        {
          "letter": "A",
          "text": "Something physical—sports, music, or movement",
          "quality": "best"
        },
        {
          "letter": "B",
          "text": "Something mental—presentations, conversations, decisions",
          "quality": "best"
        }
      ],
      "responses": {
        "A": {
          "script": "Perfect choice. Physical skills respond incredibly well to visualization. Tonight, before sleep, spend five minutes seeing yourself perform it perfectly. Feel every motion. You'll be surprised what happens.",
          "audio": "/audio/351/wisdom_response_a.mp3",
          "duration": 12
        },
        "B": {
          "script": "Excellent. Visualization works for mental skills too—public speaking, difficult conversations, high-pressure decisions. Run through the scenario. See yourself handling it with grace. Your brain will be more prepared when it's real.",
          "audio": "/audio/351/wisdom_response_b.mp3",
          "duration": 13
        }
      },
      "studentComment": {
        "name": "Taylor",
        "avatar": "👤",
        "text": "I'm going to try this before my job interview next week!",
        "audio": "/audio/351/wisdom_comment.mp3",
        "duration": 4
      }
    },
    "outro": {
      "name": "Outro",
      "icon": "👋",
      "order": 7,
      "talk": {
        "script": "That's today's lesson. Your brain is more trainable than you ever imagined—literally. Visualization isn't wishful thinking. It's cognitive rehearsal that primes your brain for performance. Tonight, give it a try. Close your eyes. Pick something you want to master. And practice it in the one gym that's always open—your mind.",
        "duration": 20,
        "audio": "/audio/351/outro_talk.mp3",
        "kellyPose": "warm",
        "kellyEmotion": "encouraging"
      },
      "question": {
        "prompt": "Will you try visualization practice tonight?",
        "audio": "/audio/351/outro_question.mp3",
        "duration": 3
      },
      "options": [
        {
          "letter": "A",
          "text": "Yes, I'm going to give it a shot!",
          "quality": "best"
        },
        {
          "letter": "B",
          "text": "Maybe—I need to think about what to practice",
          "quality": "good"
        }
      ],
      "responses": {
        "A": {
          "script": "Love that energy! Remember: specific, vivid, multi-sensory. See you tomorrow with something new. Keep visualizing great things.",
          "audio": "/audio/351/outro_response_a.mp3",
          "duration": 8
        },
        "B": {
          "script": "Take your time choosing. The right skill will come to you. When you're ready, your brain will be too. See you tomorrow!",
          "audio": "/audio/351/outro_response_b.mp3",
          "duration": 8
        }
      },
      "studentComment": {
        "name": "Casey",
        "avatar": "👤",
        "text": "Best 3 minutes I've spent today. Thanks, Kelly! ✨",
        "audio": "/audio/351/outro_comment.mp3",
        "duration": 4
      }
    }
  },
  "phaseOrder": ["hook", "cliff", "fact1", "fact2", "fact3", "wisdom", "outro"],
  "audioManifest": {
    "total_files": 35,
    "files": [
      "hook_talk.mp3", "hook_question.mp3", "hook_response_a.mp3", "hook_response_b.mp3", "hook_comment.mp3",
      "cliff_talk.mp3", "cliff_question.mp3", "cliff_response_a.mp3", "cliff_response_b.mp3", "cliff_comment.mp3",
      "fact1_talk.mp3", "fact1_question.mp3", "fact1_response_a.mp3", "fact1_response_b.mp3", "fact1_comment.mp3",
      "fact2_talk.mp3", "fact2_question.mp3", "fact2_response_a.mp3", "fact2_response_b.mp3", "fact2_comment.mp3",
      "fact3_talk.mp3", "fact3_question.mp3", "fact3_response_a.mp3", "fact3_response_b.mp3", "fact3_comment.mp3",
      "wisdom_talk.mp3", "wisdom_question.mp3", "wisdom_response_a.mp3", "wisdom_response_b.mp3", "wisdom_comment.mp3",
      "outro_talk.mp3", "outro_question.mp3", "outro_response_a.mp3", "outro_response_b.mp3", "outro_comment.mp3"
    ],
    "base_path": "/audio/351/"
  },
  "atoms": [
    {
      "id": "day351-hook",
      "phase": "Hook",
      "content": {
        "script": "Ever wondered why athletes close their eyes before a big moment? They're not just calming their nerves. They're doing something far more powerful—they're practicing. Without moving a muscle. It's called visualization, and the science behind it might change how you think about learning itself.",
        "choice_intro": "Before we dive in—have you ever imagined doing something before you actually did it?",
        "option_a": "Yes, I mentally rehearse things sometimes",
        "option_b": "Not really, I usually just wing it",
        "success_response": "You're already tapping into something powerful. Today you'll learn exactly why that works—and how to do it even better.",
        "alt_response": "That's totally normal! Most people don't realize what they're missing. By the end of today, you might change your approach.",
        "kellyPose": "curious",
        "kellyEmotion": "intrigued"
      },
      "visual_url": "/kelly/phases/351/hook.png"
    },
    {
      "id": "day351-cliff",
      "phase": "Cliff",
      "content": {
        "script": "Here's where it gets interesting. When you vividly imagine doing something—really see it, feel it, experience it in your mind—your brain activates almost the same way as when you actually do it. The neurons fire. The pathways light up. But here's the question that puzzled scientists for years...",
        "choice_intro": "Why would imagining something make you better at actually doing it?",
        "option_a": "It's probably just confidence—positive thinking",
        "option_b": "Maybe it actually trains the brain somehow",
        "success_response": "Exactly right. And not in some vague, mystical way—we're talking measurable, physical changes in neural structure. Let me show you the evidence.",
        "alt_response": "That's what researchers thought at first too! Confidence does play a role. But brain scans revealed something far more concrete happening inside the skull.",
        "kellyPose": "explaining",
        "kellyEmotion": "thoughtful"
      },
      "visual_url": "/kelly/phases/351/hook.png"
    },
    {
      "id": "day351-fact1",
      "phase": "Fact1",
      "content": {
        "script": "When you imagine performing an action, your motor cortex—that's the part of your brain that controls movement—lights up almost identically to when you actually move. Brain scans show about 90% overlap. Ninety percent. Your brain literally cannot tell the difference between vividly imagining something and doing it. It's practicing either way.",
        "choice_intro": "What do you think this means for learning new skills?",
        "option_a": "You could practice anywhere, anytime—even without equipment",
        "option_b": "It might help, but real practice is probably still way better",
        "success_response": "You've got it. On the bus, in bed, waiting in line—your brain doesn't care where your body is. It's ready to train.",
        "alt_response": "Real practice is important, absolutely. But here's the thing—the best performers don't choose one or the other. They combine both. And the results are remarkable.",
        "kellyPose": "explaining",
        "kellyEmotion": "enthusiastic",
        "factNumber": 1,
        "factTitle": "Neural Overlap"
      },
      "visual_url": "/kelly/phases/351/q1.png"
    },
    {
      "id": "day351-fact2",
      "phase": "Fact2",
      "content": {
        "script": "Let me tell you about a famous experiment. Researchers took people who had never played piano and divided them into three groups. Group one physically practiced a simple piece for five days. Group two only imagined practicing—same piece, same time, but never touched a key. Group three did nothing. After five days, they scanned everyone's brains. The results shocked the scientific community.",
        "choice_intro": "What do you think they found when comparing the imagination group to the physical practice group?",
        "option_a": "The imagination group showed some improvement, but way less",
        "option_b": "Their brains changed almost identically",
        "success_response": "Exactly. The brain regions responsible for piano playing grew in both groups. Imagination alone rewired their brains. Not as much as physical practice, but remarkably close.",
        "alt_response": "That's the logical guess. But here's the twist—the imagination group's brains showed nearly identical changes to the physical practice group. Mental rehearsal created real, measurable neuroplastic changes.",
        "kellyPose": "storytelling",
        "kellyEmotion": "engaged",
        "factNumber": 2,
        "factTitle": "The Piano Study"
      },
      "visual_url": "/kelly/phases/351/q2.png"
    },
    {
      "id": "day351-fact3",
      "phase": "Fact3",
      "content": {
        "script": "This isn't just lab science. Elite performers have known this for decades. Olympic athletes spend up to 50% of their training time on mental rehearsal. Surgeons visualize entire procedures before making a single cut. Concert pianists play through pieces in their minds on the flight to performances. The key they all discovered: specificity. Vague daydreaming doesn't work. You need vivid, detailed, multi-sensory imagination.",
        "choice_intro": "What makes visualization most effective, based on what the pros do?",
        "option_a": "Imagining success and positive outcomes",
        "option_b": "Vivid detail—seeing, feeling, hearing every step",
        "success_response": "That's the key. The more senses you engage, the more your brain treats it as real practice. See it, feel it, hear it. First-person perspective. Every detail matters.",
        "alt_response": "Positive outcomes matter for motivation, but here's the secret the pros know: you have to visualize the process, not just the result. Feel the movements. See the environment. Hear the sounds. That's what triggers the neural overlap.",
        "kellyPose": "passionate",
        "kellyEmotion": "inspired",
        "factNumber": 3,
        "factTitle": "Elite Practice"
      },
      "visual_url": "/kelly/phases/351/q3.png"
    },
    {
      "id": "day351-wisdom",
      "phase": "Wisdom",
      "content": {
        "script": "Here's today's wisdom: Your imagination is a practice field. The mind that rehearses builds pathways the passive mind never develops. Every time you vividly imagine doing something, you're laying down the neural tracks that make it easier to do for real. This is one of the few truly free performance enhancers available to every human being.",
        "choice_intro": "What's one skill you'd like to practice in your mind this week?",
        "option_a": "Something physical—sports, music, or movement",
        "option_b": "Something mental—presentations, conversations, decisions",
        "success_response": "Perfect choice. Physical skills respond incredibly well to visualization. Tonight, before sleep, spend five minutes seeing yourself perform it perfectly. Feel every motion. You'll be surprised what happens.",
        "alt_response": "Excellent. Visualization works for mental skills too—public speaking, difficult conversations, high-pressure decisions. Run through the scenario. See yourself handling it with grace. Your brain will be more prepared when it's real.",
        "kellyPose": "warm",
        "kellyEmotion": "wise"
      },
      "visual_url": "/kelly/phases/351/wisdom.png"
    },
    {
      "id": "day351-outro",
      "phase": "Outro",
      "content": {
        "script": "That's today's lesson. Your brain is more trainable than you ever imagined—literally. Visualization isn't wishful thinking. It's cognitive rehearsal that primes your brain for performance. Tonight, give it a try. Close your eyes. Pick something you want to master. And practice it in the one gym that's always open—your mind.",
        "choice_intro": "Will you try visualization practice tonight?",
        "option_a": "Yes, I'm going to give it a shot!",
        "option_b": "Maybe—I need to think about what to practice",
        "success_response": "Love that energy! Remember: specific, vivid, multi-sensory. See you tomorrow with something new. Keep visualizing great things.",
        "alt_response": "Take your time choosing. The right skill will come to you. When you're ready, your brain will be too. See you tomorrow!",
        "kellyPose": "warm",
        "kellyEmotion": "encouraging"
      },
      "visual_url": "/kelly/phases/351/wisdom.png"
    }
  ],
  "shards": [
    {
      "id": "shard-351-001",
      "type": "fun_fact",
      "content": "Brain scans show visualization activates 90% of the same neural areas as actually doing something"
    },
    {
      "id": "shard-351-002",
      "type": "fun_fact",
      "content": "Olympic athletes spend up to 50% of their training time on mental rehearsal"
    },
    {
      "id": "shard-351-003",
      "type": "fun_fact",
      "content": "Pianists who only visualized practicing improved nearly as much as those who physically practiced"
    }
  ],
  "growTrack": {
    "title": "Learning Accountability - Staying on Track",
    "emoji": "🎯",
    "learning_objective": "Create accountability structures to follow through on learning commitments",
    "activity": "Choose one person you trust and tell them about your learning goal for the week. Ask them to check in with you in 7 days."
  }
};
// Register in LOCAL_PACKS for fallback engine
window.CURIOUS_KELLY.LOCAL_PACKS[351] = window.CURIOUS_KELLY.DAY_351;
