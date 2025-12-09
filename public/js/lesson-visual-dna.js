/**
 * LESSON VISUAL DNA - 365 Unique Visual Identities
 * 
 * Each lesson gets a unique visual fingerprint based on:
 * - Primary color palette (tied to topic category)
 * - Secondary accent colors
 * - Pattern type (geometric, organic, cosmic, etc.)
 * - Icon/symbol representation
 * - Animation style
 * - Kelly's energy/mood for this topic
 */

const LESSON_VISUAL_DNA = {
    // === CATEGORY COLOR SYSTEMS ===
    categories: {
        cosmos: {
            primary: ['#0f0c29', '#302b63', '#24243e'],
            accent: ['#a855f7', '#6366f1', '#ec4899'],
            pattern: 'stardust',
            energy: 'wonder'
        },
        nature: {
            primary: ['#134e5e', '#71b280', '#0f4c3a'],
            accent: ['#10b981', '#84cc16', '#22d3ee'],
            pattern: 'organic',
            energy: 'growth'
        },
        water: {
            primary: ['#1a2980', '#26d0ce', '#0c2340'],
            accent: ['#06b6d4', '#3b82f6', '#60a5fa'],
            pattern: 'waves',
            energy: 'flow'
        },
        mind: {
            primary: ['#200122', '#6f0000', '#1a0533'],
            accent: ['#f43f5e', '#f97316', '#fbbf24'],
            pattern: 'neural',
            energy: 'spark'
        },
        emotion: {
            primary: ['#2c1654', '#c94b4b', '#4b134f'],
            accent: ['#ec4899', '#f472b6', '#fda4af'],
            pattern: 'heartbeat',
            energy: 'warm'
        },
        growth: {
            primary: ['#403a3e', '#be5869', '#2c3e50'],
            accent: ['#fbbf24', '#f59e0b', '#f97316'],
            pattern: 'sunrise',
            energy: 'inspire'
        },
        creative: {
            primary: ['#4b1248', '#f0c27b', '#5d1049'],
            accent: ['#a855f7', '#ec4899', '#f43f5e'],
            pattern: 'geometric',
            energy: 'create'
        },
        body: {
            primary: ['#2b0a3d', '#dc143c', '#1a0533'],
            accent: ['#ef4444', '#f97316', '#fca5a1'],
            pattern: 'cellular',
            energy: 'vital'
        },
        social: {
            primary: ['#232526', '#414345', '#1c1c1e'],
            accent: ['#fbbf24', '#fb923c', '#facc15'],
            pattern: 'connection',
            energy: 'together'
        },
        earth: {
            primary: ['#2c3e50', '#4a6741', '#1a3a2a'],
            accent: ['#84cc16', '#22c55e', '#a3e635'],
            pattern: 'terrain',
            energy: 'ground'
        }
    },

    // === 365 LESSONS VISUAL MAPPINGS ===
    // Format: [category, uniqueHue, patternVariant, iconEmoji, kellySpeaks]
    lessons: {
        1: ['growth', 15, 'A', '✨', "Today we begin. Together."],
        2: ['water', 200, 'B', '💧', "Water is a shapeshifter."],
        3: ['water', 210, 'A', '☁️', "Look up. The sky is telling stories."],
        4: ['cosmos', 280, 'C', '💡', "Light is the universe's messenger."],
        5: ['nature', 180, 'B', '🔊', "Sound paints pictures you can't see."],
        6: ['nature', 120, 'A', '🌱', "Inside this tiny seed? A whole forest."],
        7: ['cosmos', 270, 'B', '⭐', "You're made of star stuff. Literally."],
        8: ['emotion', 340, 'A', '🤝', "Real friends show up. Even when it's hard."],
        9: ['emotion', 350, 'B', '💝', "One act of kindness creates ripples."],
        10: ['social', 40, 'A', '👂', "The best gift? Your full attention."],
        11: ['growth', 45, 'B', '⏳', "Good things take time. That's the point."],
        12: ['emotion', 30, 'C', '🙏', "Gratitude rewires your brain."],
        13: ['mind', 300, 'A', '💭', "Your dreams are your brain's creativity lab."],
        14: ['creative', 320, 'B', '❓', "Questions are more powerful than answers."],
        15: ['mind', 10, 'A', '🧠', "Your brain is building itself. Right now."],
        16: ['creative', 330, 'C', '🎵', "Music speaks what words can't."],
        17: ['growth', 20, 'A', '💪', "Mistakes aren't failures. They're data."],
        18: ['nature', 100, 'B', '🌿', "Plants are eating sunlight. Think about that."],
        19: ['body', 0, 'A', '❤️', "Your blood tells the story of humanity."],
        20: ['body', 350, 'B', '😴', "Sleep is when the magic happens."],
        21: ['social', 50, 'C', '🗣️', "Disagreement doesn't mean disrespect."],
        22: ['cosmos', 260, 'A', '🌍', "Gravity is the universe hugging itself."],
        23: ['mind', 290, 'B', '🎯', "Boredom is creativity waiting to happen."],
        24: ['mind', 320, 'A', '🔮', "Memories aren't recordings. They're reconstructions."],
        25: ['creative', 340, 'C', '📚', "Stories are how we make sense of chaos."],
        26: ['body', 60, 'B', '😊', "Smiling tricks your brain into happiness."],
        27: ['earth', 190, 'A', '🏔️', "Ice carved the world we walk on."],
        28: ['growth', 25, 'B', '🦁', "Courage isn't fearless. It's fear + action."],
        29: ['earth', 90, 'C', '🌾', "Beneath your feet: billions of lives."],
        30: ['social', 45, 'A', '🤗', "Helping strangers helps you too."],
        31: ['growth', 35, 'B', '🔄', "Change is the only constant. Embrace it."],
        // February
        32: ['nature', 110, 'A', '🌲', "Trees are the planet's lungs."],
        33: ['cosmos', 275, 'B', '🌙', "The moon moves oceans."],
        34: ['emotion', 345, 'A', '😢', "Tears are the body's release valve."],
        35: ['mind', 305, 'C', '🎭', "Your brain has many characters."],
        36: ['water', 205, 'B', '❄️', "Every snowflake is a tiny miracle."],
        37: ['creative', 325, 'A', '🎨', "Color changes how you feel."],
        38: ['body', 5, 'B', '🫀', "Your heart beats 100,000 times a day."],
        39: ['social', 55, 'C', '👥', "We're wired for connection."],
        40: ['earth', 95, 'A', '🌋', "The earth is alive and moving."],
        41: ['growth', 40, 'B', '🎓', "Learning never stops. That's the gift."],
        42: ['nature', 115, 'A', '🦋', "Transformation takes patience."],
        43: ['cosmos', 265, 'C', '🔭', "The universe is unimaginably vast."],
        44: ['emotion', 355, 'B', '😤', "Anger is information. Listen to it."],
        45: ['mind', 310, 'A', '💡', "Creativity is connecting dots."],
        46: ['water', 215, 'B', '🌊', "The ocean remembers everything."],
        47: ['creative', 335, 'C', '✍️', "Writing is thinking made visible."],
        48: ['body', 15, 'A', '👃', "Your nose knows more than you think."],
        49: ['social', 60, 'B', '🏠', "Home is people, not places."],
        50: ['earth', 100, 'A', '🗻', "Mountains are patience made solid."],
        51: ['growth', 30, 'C', '🌟', "You're exactly where you need to be."],
        52: ['nature', 125, 'B', '🐝', "Bees hold civilization together."],
        53: ['cosmos', 285, 'A', '💫', "Black holes bend time itself."],
        54: ['emotion', 360, 'B', '🥰', "Love is a practice, not just a feeling."],
        55: ['mind', 315, 'C', '🧩', "Your brain loves patterns."],
        56: ['water', 195, 'A', '🌧️', "Rain is the sky's gift."],
        57: ['creative', 345, 'B', '🎹', "Rhythm is built into your body."],
        58: ['body', 355, 'A', '👀', "Your eyes see the world upside down."],
        59: ['social', 65, 'C', '🤝', "Trust is built in small moments."],
        // March
        60: ['earth', 105, 'B', '🌸', "Spring is earth waking up."],
        61: ['growth', 50, 'A', '🚀', "Start before you're ready."],
        62: ['nature', 130, 'C', '🦅', "Birds see magnetic fields."],
        63: ['cosmos', 290, 'B', '☄️', "Comets are time capsules."],
        64: ['emotion', 340, 'A', '😌', "Peace is a practice."],
        65: ['mind', 300, 'B', '🎲', "Randomness is underrated."],
        66: ['water', 220, 'C', '💦', "Sweat is your body's AC."],
        67: ['creative', 315, 'A', '📸', "Photos freeze moments."],
        68: ['body', 10, 'B', '🦴', "Your bones are living tissue."],
        69: ['social', 70, 'A', '👋', "First impressions can be wrong."],
        70: ['earth', 85, 'C', '🌍', "The earth spins at 1,000 mph."],
        71: ['growth', 55, 'B', '🎯', "Goals are dreams with deadlines."],
        72: ['nature', 135, 'A', '🍄', "Fungi run underground networks."],
        73: ['cosmos', 295, 'C', '🌌', "Galaxies are island universes."],
        74: ['emotion', 335, 'B', '😊', "Joy is a choice."],
        75: ['mind', 325, 'A', '🔍', "Attention is your superpower."],
        76: ['water', 225, 'B', '🏊', "Water supports you."],
        77: ['creative', 350, 'C', '🎬', "Stories have structures."],
        78: ['body', 20, 'A', '💪', "Muscles have memory."],
        79: ['social', 75, 'B', '🗨️', "Words shape reality."],
        80: ['earth', 110, 'A', '🌿', "Green soothes the soul."],
        81: ['growth', 45, 'C', '📈', "Progress isn't linear."],
        82: ['nature', 140, 'B', '🐜', "Ants build empires."],
        83: ['cosmos', 255, 'A', '🪐', "Planets dance around stars."],
        84: ['emotion', 330, 'B', '😔', "Sadness is part of wholeness."],
        85: ['mind', 330, 'C', '💭', "Thoughts aren't facts."],
        86: ['water', 230, 'A', '🧊', "Ice floats. That's weird and important."],
        87: ['creative', 355, 'B', '🎤', "Your voice is unique."],
        88: ['body', 25, 'A', '🫁', "You breathe 20,000 times a day."],
        89: ['social', 80, 'C', '❤️', "Kindness is contagious."],
        90: ['earth', 115, 'B', '🌱', "Everything grows in spring."],
        // April
        91: ['growth', 60, 'A', '🌈', "After rain comes rainbows."],
        92: ['nature', 145, 'C', '🐸', "Frogs were here before dinosaurs."],
        93: ['cosmos', 260, 'B', '☀️', "The sun is a nuclear reactor."],
        94: ['emotion', 345, 'A', '😅', "Laughter is medicine."],
        95: ['mind', 335, 'B', '🧠', "Sleep cleans your brain."],
        96: ['water', 235, 'C', '🌈', "Rainbows are light tricks."],
        97: ['creative', 360, 'A', '🖌️', "Art is communication."],
        98: ['body', 30, 'B', '👅', "Taste buds regenerate weekly."],
        99: ['social', 85, 'A', '🙌', "Celebration matters."],
        100: ['earth', 120, 'C', '🌺', "Flowers are evolution's billboards."],
        101: ['growth', 65, 'B', '⚡', "Energy follows focus."],
        102: ['nature', 150, 'A', '🕷️', "Spiders are engineers."],
        103: ['cosmos', 265, 'B', '🌑', "Eclipses unite humanity."],
        104: ['emotion', 350, 'C', '🤔', "Confusion precedes clarity."],
        105: ['mind', 340, 'A', '💡', "Aha moments change everything."],
        106: ['water', 240, 'B', '⛈️', "Storms clear the air."],
        107: ['creative', 305, 'A', '📝', "Journaling is self-discovery."],
        108: ['body', 35, 'C', '🦷', "Teeth are stronger than bone."],
        109: ['social', 90, 'B', '🎁', "Giving feels better than getting."],
        110: ['earth', 125, 'A', '🌳', "Trees talk to each other."],
        111: ['growth', 70, 'B', '🔥', "Passion fuels persistence."],
        112: ['nature', 155, 'C', '🐢', "Slow and steady works."],
        113: ['cosmos', 270, 'A', '💨', "Solar wind shapes our sky."],
        114: ['emotion', 355, 'B', '😇', "Forgiveness frees you."],
        115: ['mind', 345, 'A', '🎭', "We all wear masks."],
        116: ['water', 245, 'C', '🏖️', "Tides are moon-powered."],
        117: ['creative', 310, 'B', '🎭', "Drama reveals truth."],
        118: ['body', 40, 'A', '🩸', "Blood types tell stories."],
        119: ['social', 95, 'B', '🏆', "Success is a team sport."],
        120: ['earth', 130, 'C', '🐛', "Worms make soil alive."],
        // May
        121: ['growth', 75, 'A', '🌻', "Seek the light."],
        122: ['nature', 160, 'B', '🦜', "Birds migrate by stars."],
        123: ['cosmos', 275, 'C', '🔬', "Atoms are mostly empty space."],
        124: ['emotion', 360, 'A', '🙏', "Vulnerability is strength."],
        125: ['mind', 350, 'B', '📊', "Data tells stories."],
        126: ['water', 250, 'A', '💧', "Hydration is brain fuel."],
        127: ['creative', 315, 'C', '🎶', "Music transcends language."],
        128: ['body', 45, 'B', '🧬', "Your DNA is 99.9% same as everyone's."],
        129: ['social', 100, 'A', '🤷', "It's okay to not know."],
        130: ['earth', 135, 'B', '🦔', "Small creatures matter most."],
        131: ['growth', 80, 'C', '💫', "Believe in what you can't see yet."],
        132: ['nature', 165, 'A', '🌾', "Grass feeds the world."],
        133: ['cosmos', 280, 'B', '🌠', "Shooting stars are tiny rocks."],
        134: ['emotion', 340, 'A', '😤', "Boundaries are beautiful."],
        135: ['mind', 355, 'C', '🔄', "Habits shape destiny."],
        136: ['water', 255, 'B', '⚓', "The deep ocean is alien."],
        137: ['creative', 320, 'A', '🎪', "Play is essential."],
        138: ['body', 50, 'B', '🦠', "Microbes outnumber your cells."],
        139: ['social', 105, 'C', '🎉', "Rituals create meaning."],
        140: ['earth', 140, 'A', '🐝', "Pollination is partnership."],
        141: ['growth', 85, 'B', '🎢', "Life has seasons."],
        142: ['nature', 170, 'A', '🌴', "Palm trees survived dinosaurs."],
        143: ['cosmos', 285, 'C', '🛸', "We might not be alone."],
        144: ['emotion', 345, 'B', '😌', "Calm is a superpower."],
        145: ['mind', 360, 'A', '🎯', "Focus beats talent."],
        146: ['water', 200, 'B', '🏄', "Go with the flow."],
        147: ['creative', 325, 'C', '📖', "Every book changes you."],
        148: ['body', 55, 'A', '👂', "Balance lives in your ears."],
        149: ['social', 110, 'B', '🏡', "Community is everything."],
        150: ['earth', 145, 'A', '🌵', "Desert life is resourceful."],
        151: ['growth', 90, 'C', '🌊', "Waves of change."],
        // June
        152: ['nature', 175, 'B', '☀️', "Solstice celebrates light."],
        153: ['cosmos', 290, 'A', '🌌', "Dark matter holds galaxies together."],
        154: ['emotion', 350, 'B', '💗', "Hearts know things minds don't."],
        155: ['mind', 305, 'C', '🧩', "Curiosity is the answer."],
        156: ['water', 205, 'A', '🏞️', "Rivers carve canyons."],
        157: ['creative', 330, 'B', '🎨', "Color is vibration."],
        158: ['body', 60, 'A', '🏃', "Movement is medicine."],
        159: ['social', 115, 'C', '👪', "Family is chosen too."],
        160: ['earth', 150, 'B', '🌊', "Coral reefs are rainforests."],
        161: ['growth', 95, 'A', '⭐', "You're more capable than you think."],
        162: ['nature', 180, 'B', '🐋', "Whales sing across oceans."],
        163: ['cosmos', 295, 'C', '⏰', "Time is relative."],
        164: ['emotion', 355, 'A', '🎭', "All feelings are valid."],
        165: ['mind', 310, 'B', '💭', "Daydreaming is useful."],
        166: ['water', 210, 'A', '💨', "Humidity affects everything."],
        167: ['creative', 335, 'C', '🎬', "Life is a story."],
        168: ['body', 65, 'B', '🦴', "Bones heal stronger."],
        169: ['social', 120, 'A', '🤝', "Cooperation beats competition."],
        170: ['earth', 155, 'B', '🦎', "Adaptation is survival."],
        171: ['growth', 100, 'C', '🌈', "After storms: beauty."],
        172: ['nature', 185, 'A', '🌙', "Nocturnal life is rich."],
        173: ['cosmos', 300, 'B', '🔮', "Quantum is weird."],
        174: ['emotion', 360, 'A', '😊', "Happiness is a skill."],
        175: ['mind', 315, 'C', '📚', "Reading rewires brains."],
        176: ['water', 215, 'B', '🧊', "Glaciers are time machines."],
        177: ['creative', 340, 'A', '🎤', "Expression heals."],
        178: ['body', 70, 'B', '💤', "Dreams process emotions."],
        179: ['social', 125, 'C', '🌍', "We're one species."],
        180: ['earth', 160, 'A', '🏔️', "Mountains inspire."],
        181: ['growth', 105, 'B', '🔑', "Keys appear when ready."],
        // July
        182: ['nature', 190, 'C', '🔥', "Heat makes things happen."],
        183: ['cosmos', 305, 'A', '🚀', "Space exploration reveals us."],
        184: ['emotion', 340, 'B', '🎈', "Joy can be simple."],
        185: ['mind', 320, 'A', '🎯', "Intention matters."],
        186: ['water', 220, 'C', '🏊', "Swimming is flying."],
        187: ['creative', 345, 'B', '🎭', "Performance is presence."],
        188: ['body', 75, 'A', '🌡️', "Your body regulates perfectly."],
        189: ['social', 130, 'B', '🗳️', "Voice matters."],
        190: ['earth', 165, 'C', '🌴', "Tropical life explodes."],
        191: ['growth', 110, 'A', '💎', "Pressure makes diamonds."],
        192: ['nature', 195, 'B', '🦁', "Every creature has purpose."],
        193: ['cosmos', 310, 'A', '🌒', "Phases teach patience."],
        194: ['emotion', 345, 'C', '🫂', "Hugs are healing."],
        195: ['mind', 325, 'B', '🧘', "Stillness speaks."],
        196: ['water', 225, 'A', '⛵', "Wind and water dance."],
        197: ['creative', 350, 'B', '📸', "Perspective is everything."],
        198: ['body', 80, 'C', '🦷', "Smile science."],
        199: ['social', 135, 'A', '🎪', "Gathering is ancient."],
        200: ['earth', 170, 'B', '🌾', "Harvest rewards patience."],
        201: ['growth', 115, 'A', '🌟', "Shine anyway."],
        202: ['nature', 200, 'C', '🐘', "Memory lives in species."],
        203: ['cosmos', 315, 'B', '💫', "Energy can't be destroyed."],
        204: ['emotion', 350, 'A', '🙏', "Gratitude transforms."],
        205: ['mind', 330, 'B', '💡', "Ideas are contagious."],
        206: ['water', 230, 'C', '🌊', "Oceans regulate climate."],
        207: ['creative', 355, 'A', '🖼️', "Art outlasts empires."],
        208: ['body', 85, 'B', '👃', "Smell triggers memories."],
        209: ['social', 140, 'A', '🏆', "Winners lift others."],
        210: ['earth', 175, 'C', '🦋', "Metamorphosis is possible."],
        211: ['growth', 120, 'B', '🔄', "Restart anytime."],
        212: ['nature', 205, 'A', '🌻', "Face the sun."],
        // August
        213: ['cosmos', 320, 'B', '🌠', "Meteor showers connect us."],
        214: ['emotion', 355, 'C', '❤️', "Love is action."],
        215: ['mind', 335, 'A', '🎲', "Randomness sparks creativity."],
        216: ['water', 235, 'B', '💧', "Every drop matters."],
        217: ['creative', 360, 'A', '🎵', "Rhythm is universal."],
        218: ['body', 90, 'C', '💪', "Strength comes in many forms."],
        219: ['social', 145, 'B', '🗣️', "Listen more than speak."],
        220: ['earth', 180, 'A', '🐝', "Small things run the world."],
        221: ['growth', 125, 'B', '🌅', "Each day is new."],
        222: ['nature', 210, 'C', '🍃', "Wind carries seeds."],
        223: ['cosmos', 325, 'A', '⚡', "Lightning is electric."],
        224: ['emotion', 360, 'B', '😌', "Peace is possible."],
        225: ['mind', 340, 'A', '📖', "Stories shape identity."],
        226: ['water', 240, 'C', '🏖️', "Coastlines are edges."],
        227: ['creative', 305, 'B', '🎭', "Play is learning."],
        228: ['body', 95, 'A', '🧬', "You're unique. Literally."],
        229: ['social', 150, 'B', '🤝', "Trust builds slowly."],
        230: ['earth', 185, 'C', '🌿', "Green is calming."],
        231: ['growth', 130, 'A', '💫', "Believe in magic."],
        232: ['nature', 215, 'B', '🐺', "Wolves teach family."],
        233: ['cosmos', 330, 'A', '🛰️', "Satellites watch Earth."],
        234: ['emotion', 340, 'C', '🥺', "Vulnerability connects."],
        235: ['mind', 345, 'B', '🔮', "Intuition is data."],
        236: ['water', 245, 'A', '🌧️', "Rain nurtures."],
        237: ['creative', 310, 'B', '✏️', "Doodles are thinking."],
        238: ['body', 100, 'C', '🏃', "Endurance builds."],
        239: ['social', 155, 'A', '🎉', "Celebrate often."],
        240: ['earth', 190, 'B', '🌋', "Earth is dynamic."],
        241: ['growth', 135, 'A', '🚪', "New doors open."],
        242: ['nature', 220, 'C', '🍁', "Seasons teach letting go."],
        243: ['cosmos', 335, 'B', '🌌', "We're cosmic dust."],
        // September
        244: ['emotion', 345, 'A', '📚', "Learning is growth."],
        245: ['mind', 350, 'B', '🎯', "Focus is a muscle."],
        246: ['water', 250, 'C', '🌊', "Tides are rhythms."],
        247: ['creative', 315, 'A', '🎨', "Create daily."],
        248: ['body', 105, 'B', '👁️', "Eyes reveal souls."],
        249: ['social', 160, 'A', '👋', "Greetings matter."],
        250: ['earth', 195, 'C', '🍂', "Fall is release."],
        251: ['growth', 140, 'B', '⚖️', "Balance is dynamic."],
        252: ['nature', 225, 'A', '🦌', "Wildlife adapts."],
        253: ['cosmos', 340, 'B', '🌙', "Night has wisdom."],
        254: ['emotion', 350, 'C', '😊', "Joy is practice."],
        255: ['mind', 355, 'A', '🧩', "Problems are puzzles."],
        256: ['water', 255, 'B', '💦', "Cleansing refreshes."],
        257: ['creative', 320, 'A', '🎭', "Masks reveal truth."],
        258: ['body', 110, 'C', '🫀', "Heart intelligence."],
        259: ['social', 165, 'B', '🤗', "Empathy is skill."],
        260: ['earth', 200, 'A', '🌰', "Seeds wait for spring."],
        261: ['growth', 145, 'B', '🔥', "Passion sustains."],
        262: ['nature', 230, 'C', '🕸️', "Webs are engineering."],
        263: ['cosmos', 345, 'A', '⭐', "Stars are suns."],
        264: ['emotion', 355, 'B', '🙏', "Thankfulness heals."],
        265: ['mind', 360, 'A', '💭', "Thoughts create reality."],
        266: ['water', 200, 'C', '🌧️', "Weather connects everything."],
        267: ['creative', 325, 'B', '📝', "Writing is thinking."],
        268: ['body', 115, 'A', '💪', "Rest is training."],
        269: ['social', 170, 'B', '🏡', "Belonging matters."],
        270: ['earth', 205, 'C', '🍄', "Decomposition is creation."],
        271: ['growth', 150, 'A', '🌟', "You're enough."],
        272: ['nature', 235, 'B', '🐿️', "Preparation matters."],
        273: ['cosmos', 350, 'A', '🔭', "Wonder drives discovery."],
        274: ['emotion', 360, 'C', '❤️', "Love is renewable."],
        // October
        275: ['mind', 305, 'B', '👻', "Fear is information."],
        276: ['water', 205, 'A', '🌫️', "Mist creates mystery."],
        277: ['creative', 330, 'B', '🎃', "Creativity transforms."],
        278: ['body', 120, 'C', '🦴', "Bones support dreams."],
        279: ['social', 175, 'A', '🎭', "Costumes free us."],
        280: ['earth', 210, 'B', '🍁', "Colors are chemistry."],
        281: ['growth', 155, 'A', '🌙', "Dark has purpose."],
        282: ['nature', 240, 'C', '🦇', "Night creatures thrive."],
        283: ['cosmos', 355, 'B', '👻', "Mystery drives science."],
        284: ['emotion', 340, 'A', '😱', "Fear can protect."],
        285: ['mind', 310, 'B', '🧠', "Brain loves patterns."],
        286: ['water', 210, 'C', '🌫️', "Fog softens edges."],
        287: ['creative', 335, 'A', '🕷️', "Design is everywhere."],
        288: ['body', 125, 'B', '🫁', "Breath is life."],
        289: ['social', 180, 'A', '🕯️', "Light gathers people."],
        290: ['earth', 215, 'C', '🍂', "Decay feeds life."],
        291: ['growth', 160, 'B', '🌑', "Endings are beginnings."],
        292: ['nature', 245, 'A', '🦉', "Wisdom watches."],
        293: ['cosmos', 360, 'B', '⚫', "Darkness reveals stars."],
        294: ['emotion', 345, 'C', '😌', "Stillness speaks."],
        295: ['mind', 315, 'A', '🎲', "Chance creates."],
        296: ['water', 215, 'B', '❄️', "First frost comes."],
        297: ['creative', 340, 'A', '🎭', "Transform yourself."],
        298: ['body', 130, 'C', '💀', "Bones tell stories."],
        299: ['social', 185, 'B', '🕸️', "Connections matter."],
        300: ['earth', 220, 'A', '🌾', "Harvest gratitude."],
        301: ['growth', 165, 'B', '🔮', "Future is unwritten."],
        302: ['nature', 250, 'C', '🦔', "Prepare for winter."],
        303: ['cosmos', 305, 'A', '🌙', "Moon guides cycles."],
        304: ['emotion', 350, 'B', '🎃', "Fun is essential."],
        305: ['mind', 320, 'A', '👁️', "See beyond surface."],
        // November
        306: ['water', 220, 'C', '🌧️', "Gray days matter."],
        307: ['creative', 345, 'B', '🎨', "Create through darkness."],
        308: ['body', 135, 'A', '🤧', "Immune systems fight."],
        309: ['social', 190, 'B', '🦃', "Gratitude gathers."],
        310: ['earth', 225, 'C', '🍁', "Trees sleep standing."],
        311: ['growth', 170, 'A', '🙏', "Thanks transforms."],
        312: ['nature', 255, 'B', '🐻', "Rest is wisdom."],
        313: ['cosmos', 310, 'A', '🌠', "Meteor showers inspire."],
        314: ['emotion', 355, 'C', '🤗', "Warmth matters."],
        315: ['mind', 325, 'B', '📊', "Reflection reveals."],
        316: ['water', 225, 'A', '❄️', "Ice preserves."],
        317: ['creative', 350, 'B', '✍️', "Write gratitude."],
        318: ['body', 140, 'C', '🛁', "Self-care sustains."],
        319: ['social', 195, 'A', '👨‍👩‍👧', "Family is flexible."],
        320: ['earth', 230, 'B', '🌲', "Evergreens persist."],
        321: ['growth', 175, 'A', '🙏', "Abundance mindset."],
        322: ['nature', 200, 'C', '🦫', "Build for future."],
        323: ['cosmos', 315, 'B', '🌌', "Universe is home."],
        324: ['emotion', 360, 'A', '❤️', "Love endures."],
        325: ['mind', 330, 'B', '📚', "Learning never ends."],
        326: ['water', 230, 'C', '💧', "Water is life."],
        327: ['creative', 355, 'A', '🎶', "Music heals."],
        328: ['body', 145, 'B', '😴', "Rest deeply."],
        329: ['social', 200, 'A', '🤝', "Community sustains."],
        330: ['earth', 235, 'B', '🌑', "Darkness has purpose."],
        331: ['growth', 180, 'C', '✨', "Small things matter."],
        332: ['nature', 205, 'A', '🌨️', "Snow blankets."],
        333: ['cosmos', 320, 'B', '⭐', "Stars guide."],
        334: ['emotion', 340, 'A', '🙏', "Thanks changes you."],
        335: ['mind', 335, 'C', '💡', "Ideas spark."],
        // December
        336: ['water', 235, 'B', '❄️', "Snowflakes are unique."],
        337: ['creative', 360, 'A', '🎄', "Traditions anchor."],
        338: ['body', 150, 'B', '🤧', "Winter wellness."],
        339: ['social', 205, 'C', '🎁', "Giving is receiving."],
        340: ['earth', 240, 'A', '🌲', "Green in gray."],
        341: ['growth', 185, 'B', '🌟', "Light returns."],
        342: ['nature', 210, 'A', '🐧', "Adapt to thrive."],
        343: ['cosmos', 325, 'C', '🌙', "Winter solstice wisdom."],
        344: ['emotion', 345, 'B', '🎄', "Joy is choice."],
        345: ['mind', 340, 'A', '🎁', "Presence over presents."],
        346: ['water', 240, 'B', '❄️', "Crystals form."],
        347: ['creative', 305, 'C', '🎶', "Carols connect."],
        348: ['body', 155, 'A', '☕', "Warmth from within."],
        349: ['social', 210, 'B', '👨‍👩‍👧‍👦', "Gathering heals."],
        350: ['earth', 245, 'A', '🌨️', "Earth sleeps."],
        351: ['growth', 190, 'C', '🔮', "Year in review."],
        352: ['nature', 215, 'B', '🦌', "Life persists."],
        353: ['cosmos', 330, 'A', '⭐', "New stars born."],
        354: ['emotion', 350, 'B', '❤️', "Love is enough."],
        355: ['mind', 345, 'C', '📝', "Reflect deeply."],
        356: ['water', 245, 'A', '💧', "Flow forward."],
        357: ['creative', 310, 'B', '🎨', "Create your year."],
        358: ['body', 160, 'A', '🛁', "Rest and renew."],
        359: ['social', 215, 'C', '🥂', "Celebrate together."],
        360: ['earth', 250, 'B', '🌍', "Earth continues."],
        361: ['growth', 195, 'A', '💫', "Transformation awaits."],
        362: ['nature', 220, 'B', '🌱', "Seeds sleep."],
        363: ['cosmos', 335, 'C', '🌌', "Infinite possibility."],
        364: ['emotion', 355, 'A', '🙏', "Gratitude always."],
        365: ['growth', 200, 'B', '🔄', "The cycle continues."]
    },

    // === AGE-ADAPTIVE KELLY STYLES ===
    kellyStyles: {
        // Young learners (5-7)
        young: {
            avatarStyle: 'playful',
            bgEnergy: 'bouncy',
            lineWeight: 'thick',
            cornerRadius: 32,
            fontSize: 'large',
            animation: 'wiggle',
            kellyPose: 'waving',
            speechBubble: 'cloud',
            colors: { saturation: 100, brightness: 95 }
        },
        // Kids (8-12)
        kids: {
            avatarStyle: 'friendly',
            bgEnergy: 'dynamic',
            lineWeight: 'medium',
            cornerRadius: 24,
            fontSize: 'medium-large',
            animation: 'float',
            kellyPose: 'pointing',
            speechBubble: 'rounded',
            colors: { saturation: 85, brightness: 90 }
        },
        // Teens (13-17)
        teens: {
            avatarStyle: 'cool',
            bgEnergy: 'smooth',
            lineWeight: 'medium',
            cornerRadius: 16,
            fontSize: 'medium',
            animation: 'slide',
            kellyPose: 'confident',
            speechBubble: 'modern',
            colors: { saturation: 70, brightness: 85 }
        },
        // Young adults (18-35)
        youngAdults: {
            avatarStyle: 'professional',
            bgEnergy: 'subtle',
            lineWeight: 'fine',
            cornerRadius: 12,
            fontSize: 'readable',
            animation: 'fade',
            kellyPose: 'thoughtful',
            speechBubble: 'minimal',
            colors: { saturation: 60, brightness: 80 }
        },
        // Adults (36-60)
        adults: {
            avatarStyle: 'refined',
            bgEnergy: 'calm',
            lineWeight: 'fine',
            cornerRadius: 8,
            fontSize: 'comfortable',
            animation: 'subtle',
            kellyPose: 'warm',
            speechBubble: 'elegant',
            colors: { saturation: 55, brightness: 75 }
        },
        // Seniors (61+)
        seniors: {
            avatarStyle: 'warm',
            bgEnergy: 'gentle',
            lineWeight: 'medium',
            cornerRadius: 16,
            fontSize: 'large',
            animation: 'gentle',
            kellyPose: 'welcoming',
            speechBubble: 'classic',
            colors: { saturation: 50, brightness: 80 }
        }
    },

    // === TONE MODIFIERS ===
    toneModifiers: {
        curious: {
            accentShift: 0,
            energyBoost: 1.1,
            iconStyle: '🔍',
            animationSpeed: 1.0
        },
        playful: {
            accentShift: 30,
            energyBoost: 1.3,
            iconStyle: '🎮',
            animationSpeed: 1.2
        },
        serious: {
            accentShift: -15,
            energyBoost: 0.8,
            iconStyle: '📚',
            animationSpeed: 0.8
        }
    },

    // === HELPER FUNCTIONS ===
    getAgeGroup(age) {
        if (age <= 7) return 'young';
        if (age <= 12) return 'kids';
        if (age <= 17) return 'teens';
        if (age <= 35) return 'youngAdults';
        if (age <= 60) return 'adults';
        return 'seniors';
    },

    getLessonVisuals(dayNumber, settings = {}) {
        const { age = 25, tone = 'curious', language = 'en' } = settings;
        const lessonData = this.lessons[dayNumber];
        
        if (!lessonData) {
            console.warn(`No visual data for day ${dayNumber}`);
            return null;
        }

        const [category, hue, patternVariant, icon, kellySpeaks] = lessonData;
        const categoryData = this.categories[category];
        const ageGroup = this.getAgeGroup(age);
        const kellyStyle = this.kellyStyles[ageGroup];
        const toneData = this.toneModifiers[tone];

        // Calculate adjusted hue based on tone
        const adjustedHue = (hue + toneData.accentShift + 360) % 360;

        return {
            dayNumber,
            category,
            icon,
            kellySpeaks,
            patternVariant,
            colors: {
                primary: categoryData.primary,
                accent: categoryData.accent,
                hue: adjustedHue,
                saturation: kellyStyle.colors.saturation,
                brightness: kellyStyle.colors.brightness
            },
            pattern: categoryData.pattern,
            energy: categoryData.energy,
            kellyStyle,
            tone: toneData,
            ageGroup
        };
    }
};

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LESSON_VISUAL_DNA;
}




