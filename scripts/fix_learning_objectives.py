"""
Fix Learning Objectives - Generate correct objectives from titles

The titles are correct. The learning_objective fields got scrambled.
This script generates proper learning objectives based on the titles.
"""

import json
from pathlib import Path

# Map of title patterns to learning objective templates
# These create simple, clear objectives that match the title topic
OBJECTIVE_TEMPLATES = {
    # SPECIFIC TITLES FIRST (exact matches for problematic ones)
    "starting fresh": "New beginnings offer opportunities for growth and change.",
    "curious people learn more": "Curiosity drives deeper learning and understanding.",
    "energy changes form": "Energy transforms from one form to another but never disappears.",
    "rainbows form": "Rainbows form when sunlight refracts through water droplets.",
    "electricity flows": "Electricity powers our modern world through the flow of electrons.",
    "your immune system": "The immune system protects the body from disease.",
    "training your immune": "Vaccines and exposure train the immune system to fight diseases.",
    "earth's hidden treasures": "Earth contains valuable minerals, gems, and resources underground.",
    "the gas you don't notice": "Nitrogen makes up most of the air we breathe.",
    "your brain does": "The brain controls our thoughts, movements, and memories.",
    "your body's framework": "The skeleton provides structure and support for the body.",
    "what reading does": "Reading exercises the brain and builds knowledge.",
    "shortcut for repeated adding": "Multiplication is repeated addition that saves time.",
    "splitting things fairly": "Division distributes quantities equally among groups.",
    "parts of a whole": "Fractions represent parts of a whole.",
    "how we know which way": "Navigation tools and landmarks help us find our way.",
    "tools changed us": "Tools extended human capabilities and transformed civilization.",
    "invention of the wheel": "The wheel revolutionized transportation and machinery.",
    "levers multiply force": "Levers use leverage to multiply force and move heavy objects.",
    "gears transfer motion": "Gears transmit motion and change speed or direction.",
    "how ai learns": "Artificial intelligence learns patterns from large amounts of data.",
    "tv shows pictures": "Television transmits moving images through electronic signals.",
    "why we dream": "Dreams may help process emotions and consolidate memories.",
    "farming changed everything": "Agriculture allowed humans to settle and build civilizations.",
    "cold-blooded survivors": "Cold-blooded animals rely on their environment to regulate temperature.",
    "every culture dances": "Dance is a universal form of human expression and celebration.",
    "clothes communicate": "Clothing communicates identity, status, and culture.",
    "why we compete": "Competition drives improvement and reveals our capabilities.",
    "sounds agree": "Harmony occurs when musical notes sound pleasing together.",
    "your voice as an instrument": "The human voice produces sound through vocal cord vibrations.",
    "finding what was already there": "Discovery reveals what exists but was previously unknown.",
    "shape with no corners": "Circles have no corners and represent infinity and wholeness.",
    "strongest shape": "Triangles distribute force efficiently and are structurally strong.",
    "what's beyond earth": "Space contains planets, stars, galaxies, and vast distances.",
    "difference catches your eye": "Contrast makes things stand out and captures attention.",
    "trusting yourself": "Self-trust comes from knowing your values and capabilities.",
    "what fair really means": "Fairness means treating everyone with equal consideration.",
    "making things right": "Repairing harm restores relationships and trust.",
    "working together": "Collaboration combines different strengths for better outcomes.",
    "when everything changes fast": "Rapid change requires adaptation and resilience.",
    "when species disappear": "Extinction removes species from Earth permanently.",
    "how your brain changes": "The brain changes and adapts throughout life through neuroplasticity.",
    "force spread over area": "Pressure is force distributed across a surface area.",
    "speeding up and slowing down": "Acceleration is the rate of change in speed.",
    "when things come apart": "Decomposition breaks matter into simpler components.",
    "using things again": "Recycling transforms waste into new useful materials.",
    "how plants eat sunlight": "Photosynthesis converts sunlight into food for plants.",
    "from one cell to trillions": "Cell division multiplies cells to build complex organisms.",
    "hiding in plain sight": "Camouflage helps animals blend into their environment.",
    "earth's different zones": "Climate zones divide Earth into regions with distinct weather patterns.",
    "how life makes more life": "Reproduction creates new generations of living things.",
    "places species call home": "Habitats provide the conditions species need to survive.",
    "things that grow back": "Regeneration allows some organisms to regrow lost parts.",
    "ancient sunlight stored": "Fossil fuels contain energy from ancient sunlight stored in plants.",
    "when practice becomes skill": "Mastery develops when repeated practice becomes automatic.",
    "learning from getting it wrong": "Mistakes provide valuable feedback for improvement.",
    "different ways of being smart": "Intelligence takes many forms beyond traditional measures.",
    "natural abilities and practice": "Talent grows stronger through dedicated practice.",
    "asking nature questions": "Scientific inquiry investigates how nature works.",
    "explanations that keep working": "Theories are explanations supported by consistent evidence.",
    "brain and body working together": "Coordination integrates mental and physical abilities.",
    "ready before it happens": "Preparation helps us respond effectively to challenges.",
    "stopping hurt before it starts": "Prevention avoids problems before they occur.",
    "asking and respecting the answer": "Consent requires asking permission and accepting the response.",
    "who you are": "Identity encompasses our values, experiences, and sense of self.",
    "what makes you you": "Individuality comes from our unique combination of traits and experiences.",
    "being part of a place": "Belonging connects us to communities and locations.",
    "your voice in decisions": "Participation gives everyone influence in shared decisions.",
    "accepting what's given": "Receiving gracefully honors the generosity of others.",
    "the voice in your head": "Self-talk shapes how we think and feel about ourselves.",
    "words that shape beliefs": "Affirmations are positive statements that influence our mindset.",
    "being where you are": "Presence means focusing attention on the current moment.",
    "deciding right from wrong": "Moral reasoning guides ethical decision-making.",
    "looking back to learn": "Reflection examines past experiences to gain wisdom.",
    "365 days of growing": "A year of daily learning adds up to remarkable growth.",
    
    # More specific fixes for remaining issues
    "rivers shape the land": "Rivers carve landscapes through erosion over millions of years.",
    "how rivers shape": "Rivers carve landscapes through erosion over millions of years.",
    "another way to write fractions": "Decimals express fractions using powers of ten.",
    "showing who you are": "Self-expression communicates our unique identity to others.",
    "who you are when no one's looking": "Character is revealed by choices made in private.",
    "brain and body working together": "Mind-body coordination integrates mental and physical abilities.",
    "knowing who you are": "Self-awareness means understanding your own thoughts and feelings.",
    
    # Nature & Science
    "sun": "The Sun provides energy for almost all life on Earth.",
    "moon": "The Moon affects ocean tides and has inspired countless stories.",
    "star": "Stars are distant suns that light up our night sky.",
    "cloud": "Clouds form when water vapor rises and cools in the atmosphere.",
    "rain": "Rain replenishes water sources and sustains ecosystems.",
    "water": "Water is essential for all life and exists in three states.",
    "ice": "Ice forms when water freezes and floats because it's less dense than liquid water.",
    "snow": "Snow forms when water vapor freezes into ice crystals in clouds.",
    "wind": "Wind is moving air caused by differences in air pressure.",
    "thunder": "Thunder is the sound of air rapidly expanding from lightning's heat.",
    "lightning": "Lightning is a massive electrical discharge during storms.",
    "rainbow": "Rainbows form when sunlight refracts through water droplets.",
    "season": "Seasons change due to Earth's tilt as it orbits the Sun.",
    "day and night": "Day and night occur as Earth rotates on its axis.",
    "shadow": "Shadows form when objects block light.",
    "light": "Light travels in straight lines and allows us to see the world.",
    "sound": "Sound travels as vibrations through air, water, and solid materials.",
    "echo": "Echoes are sounds that bounce back from surfaces.",
    "wave": "Waves carry energy through water, air, and other materials.",
    "bubble": "Bubbles are pockets of air surrounded by thin liquid films.",
    "crystal": "Crystals form when atoms arrange in repeating patterns.",
    "fossil": "Fossils preserve ancient life and tell Earth's history.",
    "dinosaur": "Dinosaurs dominated Earth for over 160 million years.",
    "volcano": "Volcanoes release molten rock from deep within Earth.",
    "earthquake": "Earthquakes occur when Earth's tectonic plates shift.",
    "mountain": "Mountains form over millions of years through geological forces.",
    "ocean": "Oceans cover over 70% of Earth's surface and teem with life.",
    "river": "Rivers carve landscapes and provide water for civilizations.",
    "lake": "Lakes are bodies of freshwater formed in basins on land.",
    "desert": "Deserts receive very little rainfall but host adapted life forms.",
    "forest": "Forests produce oxygen and provide homes for countless species.",
    "jungle": "Jungles are dense tropical forests with incredible biodiversity.",
    "coral": "Coral reefs are built by tiny animals and support diverse ecosystems.",
    "cave": "Caves form through erosion and contain unique ecosystems.",
    "island": "Islands are landmasses surrounded by water.",
    "grass": "Grasses cover vast areas and support many ecosystems.",
    "wetland": "Wetlands filter water and provide habitat for wildlife.",
    "soil": "Soil is the foundation for plant growth and terrestrial life.",
    "rock": "Rocks tell the story of Earth's geological history.",
    "gem": "Gems are minerals valued for their beauty and rarity.",
    "metal": "Metals are elements that conduct heat and electricity.",
    "air": "Air is a mixture of gases that surrounds Earth.",
    "oxygen": "Oxygen is essential for most life forms to breathe.",
    "carbon": "Carbon is the building block of all living things.",
    "hydrogen": "Hydrogen is the simplest and most abundant element.",
    "atom": "Atoms are the basic building blocks of all matter.",
    "molecule": "Molecules are atoms bonded together.",
    "cell": "Cells are the basic units of all living things.",
    "dna": "DNA contains the instructions for building living organisms.",
    
    # Living Things
    "plant": "Plants create their own food using sunlight through photosynthesis.",
    "tree": "Trees are the largest and longest-living plants on Earth.",
    "flower": "Flowers attract pollinators and produce seeds for new plants.",
    "seed": "Seeds hold the potential for entire forests and gardens.",
    "leaf": "Leaves capture sunlight and produce food for plants.",
    "mushroom": "Mushrooms are fungi that help decompose organic matter.",
    "bacteria": "Bacteria are microscopic organisms found everywhere on Earth.",
    "insect": "Insects are the most diverse group of animals on Earth.",
    "bird": "Birds are the only animals with feathers and most can fly.",
    "fish": "Fish breathe underwater using gills and come in countless varieties.",
    "mammal": "Mammals feed their young with milk and have hair or fur.",
    "reptile": "Reptiles are cold-blooded and most lay eggs on land.",
    "amphibian": "Amphibians live both in water and on land during their lives.",
    "pet": "Pets provide companionship and teach responsibility.",
    "animal": "Animals are living things that move and respond to their environment.",
    
    # Human Body
    "brain": "The brain controls our thoughts, movements, and memories.",
    "heart": "The heart pumps blood throughout the body every moment of life.",
    "bone": "Bones provide structure and protect our vital organs.",
    "muscle": "Muscles contract and relax to create all body movements.",
    "blood": "Blood carries oxygen and nutrients to every cell in the body.",
    "skin": "Skin is our largest organ and protects us from the environment.",
    "eye": "Eyes detect light and allow us to see the world in color.",
    "ear": "Ears detect sound waves and help us maintain balance.",
    "nose": "The nose detects smells and filters the air we breathe.",
    "taste": "Taste helps us identify safe and nutritious foods.",
    "touch": "Touch allows us to feel texture, temperature, and pressure.",
    "smell": "Smell is closely linked to memory and emotion.",
    "lung": "Lungs take in oxygen and release carbon dioxide.",
    "breath": "Breathing sustains every living moment of our lives.",
    "sleep": "Sleep allows the body and mind to rest and repair.",
    "dream": "Dreams occur during sleep and may help process memories.",
    "digest": "Digestion breaks down food into nutrients the body can use.",
    "immune": "The immune system protects the body from disease.",
    
    # Emotions & Character
    "friend": "Friendship connects people across differences and distances.",
    "kindness": "Kindness ripples outward, affecting more people than we see.",
    "listen": "Listening helps us understand others and ourselves better.",
    "patience": "Patience rewards those who wait for the right moment.",
    "gratitude": "Gratitude grounds us in appreciation for what we have.",
    "courage": "Courage acts even when fear is present.",
    "curious": "Curiosity explores the unknown and asks important questions.",
    "balance": "Balance stabilizes our bodies, minds, and lives.",
    "emotion": "Emotions are complex responses that guide our behavior.",
    "happy": "Happiness comes from connections, purpose, and gratitude.",
    "sad": "Sadness is a natural response to loss and disappointment.",
    "fear": "Fear protects us by alerting us to potential dangers.",
    "anger": "Anger signals that something feels unfair or threatening.",
    "surprise": "Surprise occurs when something unexpected happens.",
    "love": "Love creates deep bonds between people and living things.",
    "empathy": "Empathy allows us to understand and share others' feelings.",
    "hope": "Hope sustains us through difficult times.",
    "trust": "Trust is built through consistent, honest actions over time.",
    "forgive": "Forgiveness releases resentment and allows healing.",
    
    # Learning & Thinking
    "memory": "Memory preserves our experiences and shapes who we are.",
    "imagination": "Imagination creates possibilities that don't yet exist.",
    "question": "Questions open doors to new understanding.",
    "story": "Stories teach lessons and preserve history.",
    "music": "Music moves our emotions and brings people together.",
    "art": "Art expresses ideas and emotions in visual form.",
    "color": "Colors express emotions and meanings across cultures.",
    "pattern": "Patterns repeat in nature, art, and mathematics.",
    "number": "Numbers help us measure, count, and understand patterns.",
    "add": "Addition combines quantities to find totals.",
    "subtract": "Subtraction finds the difference between quantities.",
    "multiply": "Multiplication is repeated addition and helps us scale quantities.",
    "divide": "Division splits quantities into equal parts.",
    "fraction": "Fractions represent parts of a whole.",
    "percent": "Percentages express amounts as parts of one hundred.",
    "measure": "Measurement uses standard units to quantify the world.",
    "shape": "Shapes are the building blocks of geometry.",
    "language": "Language allows humans to share complex ideas.",
    "write": "Writing preserves thoughts for future generations.",
    "read": "Reading opens windows into other minds and worlds.",
    "communicate": "Communication connects minds through words and gestures.",
    
    # Time & Change
    "time": "Time passes at a constant rate, yet feels different in different moments.",
    "change": "Change is constant and inevitable in all things.",
    "grow": "Growth requires change, challenge, and time.",
    "age": "Aging is a natural process of change over time.",
    "history": "History records the events that shaped our world.",
    "future": "The future is shaped by decisions we make today.",
    "cycle": "Cycles repeat in nature, from seasons to life stages.",
    
    # Society & Culture
    "family": "Families come in many forms and provide support and belonging.",
    "community": "Communities are groups of people who share space and values.",
    "culture": "Cultures are shared ways of life passed through generations.",
    "tradition": "Traditions connect us to our past and to each other.",
    "celebration": "Celebrations mark important moments and bring people together.",
    "holiday": "Holidays honor events, people, or seasonal changes.",
    "cook": "Cooking transforms ingredients into nourishing meals.",
    "farm": "Farming produces food and has shaped human civilization.",
    "tool": "Tools extend human capabilities and solve practical problems.",
    "cloth": "Clothing protects us and expresses identity and culture.",
    "home": "Homes provide shelter and reflect cultural values.",
    "city": "Cities concentrate people, resources, and opportunities.",
    "village": "Villages are small communities often close to nature.",
    "country": "Countries are political divisions with unique cultures.",
    "map": "Maps represent our world and help us navigate.",
    "explore": "Exploration has driven human discovery across land and sea.",
    
    # Technology & Invention
    "invent": "Inventions solve problems and change how we live.",
    "wheel": "The wheel revolutionized transportation and machinery.",
    "print": "The printing press made books accessible to everyone.",
    "telescope": "The telescope revealed the vastness of the universe.",
    "microscope": "The microscope unveiled the hidden world of tiny organisms.",
    "steam": "The steam engine powered the Industrial Revolution.",
    "electric": "Electricity powers our modern world and exists in nature.",
    "light bulb": "The light bulb extended productive hours beyond daylight.",
    "telephone": "The telephone connected voices across great distances.",
    "airplane": "The airplane made human flight a reality.",
    "computer": "Computers process information at incredible speeds.",
    "robot": "Robots perform tasks automatically and can assist humans.",
    "ai": "Artificial intelligence learns patterns from data.",
    "internet": "The internet connects billions of people worldwide.",
    "photo": "Photography captures moments in time.",
    "movie": "Movies tell stories through moving images.",
    "tv": "Television brings images and stories into homes.",
    "radio": "Radio transmits sound through invisible waves.",
    "medicine": "Medicine helps the body heal from illness and injury.",
    "vaccine": "Vaccines train the immune system to fight diseases.",
    "surgery": "Surgery repairs the body from inside.",
    "hygiene": "Hygiene practices prevent disease and promote health.",
    
    # Physics & Chemistry
    "gravity": "Gravity pulls objects toward each other and keeps us on Earth.",
    "magnet": "Magnets attract certain metals and have invisible fields of force.",
    "energy": "Energy transforms from one form to another but never disappears.",
    "force": "Forces push, pull, and change how objects move.",
    "motion": "Motion is the change in position of an object over time.",
    "speed": "Speed measures how fast something moves.",
    "pressure": "Pressure is force applied over an area.",
    "temperature": "Temperature measures how hot or cold something is.",
    "heat": "Heat is energy that flows from warmer to cooler objects.",
    "fire": "Fire transforms matter and has shaped human civilization.",
    "mirror": "Mirrors reflect light and show us images of ourselves.",
    
    # Math & Logic
    "geometry": "Geometry is the mathematics of shapes and space.",
    "probability": "Probability measures how likely events are to occur.",
    "statistic": "Statistics helps us find patterns in data.",
    "logic": "Logic uses rules to reach valid conclusions.",
    "reason": "Reasoning uses evidence to draw conclusions.",
    "evidence": "Evidence supports conclusions with facts and data.",
    "hypothesis": "A hypothesis is an educated guess that can be tested.",
    "experiment": "Experiments test ideas through controlled trials.",
    "observe": "Observation gathers information through careful attention.",
    "compare": "Comparison identifies similarities and differences.",
    "classify": "Classification organizes things into categories.",
    
    # Health & Safety
    "exercise": "Exercise strengthens the body and improves health.",
    "nutrition": "Nutrition provides the body with necessary nutrients.",
    "stress": "Stress is the body's response to challenges and demands.",
    "relax": "Relaxation reduces stress and restores calm.",
    "mindful": "Mindfulness focuses attention on the present moment.",
    "meditat": "Meditation trains the mind to be calm and focused.",
    "stretch": "Stretching improves flexibility and prevents injury.",
    "posture": "Posture affects health and how others perceive us.",
    "safety": "Safety practices prevent accidents and injuries.",
    "first aid": "First aid provides immediate care for injuries.",
    "emergency": "Emergency preparedness helps us respond to crises.",
    "fire safety": "Fire safety prevents fires and protects people.",
    "water safety": "Water safety prevents drowning and water accidents.",
    "internet safety": "Internet safety protects personal information online.",
    
    # Values & Ethics
    "honest": "Honesty builds trust through truthful communication.",
    "respect": "Respect acknowledges the worth and dignity of others.",
    "responsible": "Responsibility means being accountable for our actions.",
    "fair": "Fairness treats everyone with equal consideration.",
    "equal": "Equality gives all people the same rights and opportunities.",
    "justice": "Justice ensures fair treatment and consequences.",
    "right": "Rights are entitlements that protect human dignity.",
    "citizen": "Citizenship involves participating in community.",
    "democracy": "Democracy gives people a voice in governance.",
    "vote": "Voting allows citizens to choose their representatives.",
    "law": "Laws are rules that govern behavior in society.",
    "freedom": "Freedom allows people to make choices and express themselves.",
    "peace": "Peace is the absence of conflict and presence of harmony.",
    "conflict": "Conflict arises from disagreement or competing interests.",
    "negotiat": "Negotiation finds mutually acceptable solutions.",
    "consent": "Consent means freely agreeing without pressure.",
    "boundary": "Boundaries protect our physical and emotional space.",
    "privacy": "Privacy protects personal information and space.",
    "identity": "Identity is our sense of who we are.",
    "diversity": "Diversity enriches communities with different perspectives.",
    
    # Goals & Growth
    "goal": "Goals give direction and purpose to our efforts.",
    "dream": "Dreams inspire us to imagine better futures.",
    "determination": "Determination pushes us to overcome obstacles.",
    "practice": "Practice improves skills through repetition.",
    "mistake": "Mistakes are opportunities to learn and grow.",
    "mentor": "Mentors guide us with their experience and wisdom.",
    "teach": "Teaching shares knowledge and helps others grow.",
    "wisdom": "Wisdom applies knowledge to make good judgments.",
    "legacy": "Legacy is what we leave behind for future generations.",
    "service": "Service helps others without expectation of reward.",
    "value": "Values are principles that guide our choices.",
    "ethic": "Ethics are moral principles that govern behavior.",
    "character": "Character is the collection of qualities that define a person.",
    "purpose": "Purpose gives meaning and direction to life.",
    "passion": "Passion is intense enthusiasm for something meaningful.",
    "inspiration": "Inspiration sparks creativity and enthusiasm.",
    "motivation": "Motivation drives us to take action toward goals.",
    "growth mindset": "Growth mindset believes abilities can develop through effort.",
    "self-talk": "Self-talk is the internal dialogue we have with ourselves.",
    "affirm": "Affirmations are positive statements that build confidence.",
    "visualiz": "Visualization imagines desired outcomes to improve performance.",
    
    # Trade & Money
    "trade": "Trade exchanges goods and services between people.",
    "money": "Money is a medium of exchange that stores value.",
    "save": "Saving sets aside resources for future use.",
    "spend": "Spending exchanges money for goods and services.",
    "give": "Giving shares resources to help others.",
    "work": "Work creates value through effort and skill.",
}

def generate_objective(title):
    """Generate a learning objective based on the title."""
    title_lower = title.lower()
    
    # Check for WORD BOUNDARY matches (not substring matches)
    import re
    for keyword, objective in OBJECTIVE_TEMPLATES.items():
        # Match keyword as a whole word, not as part of another word
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, title_lower):
            return objective
    
    # Fallback: Create a generic but relevant objective
    # Extract the main concept from the title
    if title.startswith("How "):
        topic = title[4:].rstrip("?")
        return f"Understanding how {topic.lower()} helps us make sense of the world."
    elif title.startswith("Why "):
        topic = title[4:].rstrip("?")
        return f"Understanding why {topic.lower()} matters to us and the world."
    elif title.startswith("What "):
        topic = title[5:].rstrip("?")
        return f"Understanding what {topic.lower()} reveals about our world."
    elif title.startswith("The "):
        topic = title[4:]
        return f"{topic} plays an important role in how the world works."
    elif title.startswith("When "):
        topic = title[5:]
        return f"Understanding when {topic.lower()} shaped history and our lives."
    elif title.startswith("Where "):
        topic = title[6:]
        return f"Understanding where {topic.lower()} and its significance."
    else:
        # Use the title itself as a base
        return f"{title} is an important concept that helps us understand our world."

def fix_curriculum():
    """Fix learning objectives in all curriculum files."""
    print("")
    print("=" * 70)
    print("   FIXING LEARNING OBJECTIVES")
    print("=" * 70)
    print("")
    
    # Load the calendar
    calendar_path = Path(__file__).parent.parent / "lessons" / "365_day_calendar.json"
    with open(calendar_path, "r", encoding="utf-8") as f:
        calendar_data = json.load(f)
    
    lessons = calendar_data.get("lessons", [])
    print(f"Processing {len(lessons)} lessons...")
    print("")
    
    fixed_count = 0
    
    for lesson in lessons:
        title = lesson.get("title", "")
        old_objective = lesson.get("learning_objective", "")
        new_objective = generate_objective(title)
        
        if old_objective != new_objective:
            lesson["learning_objective"] = new_objective
            fixed_count += 1
    
    # Save updated calendar
    calendar_data["fixed_at"] = "2025-12-16"
    calendar_data["fix_note"] = "Learning objectives regenerated from titles"
    
    with open(calendar_path, "w", encoding="utf-8") as f:
        json.dump(calendar_data, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed {fixed_count} learning objectives in calendar")
    
    # Now regenerate the monthly curriculum files
    print("")
    print("Regenerating monthly curriculum files...")
    
    # Import and run the generation script
    import subprocess
    result = subprocess.run(
        ["python", str(Path(__file__).parent / "generate_learn_track_from_supabase.py")],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    
    print("")
    print("=" * 70)
    print(f"   COMPLETE: Fixed {fixed_count} objectives")
    print("=" * 70)
    print("")
    
    # Show samples
    print("Sample fixes:")
    print("-" * 70)
    for lesson in lessons[56:62]:  # Days 57-62 which had major issues
        print(f"Day {lesson['day']}: {lesson['title']}")
        print(f"  -> {lesson['learning_objective']}")
        print("")

if __name__ == "__main__":
    fix_curriculum()
