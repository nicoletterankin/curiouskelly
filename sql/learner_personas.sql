-- ═══════════════════════════════════════════════════════════════════
-- LEARNER PERSONAS TABLE
-- Rich persona data for social learning simulation
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS learner_personas (
  id TEXT PRIMARY KEY,                    -- 'emma-us', 'yuki-jp', etc.
  
  -- Basic Info
  name TEXT NOT NULL,
  age INT NOT NULL,
  country_code TEXT NOT NULL,             -- 'US', 'JP', 'BR'
  country_flag TEXT NOT NULL,             -- '🇺🇸', '🇯🇵', '🇧🇷'
  
  -- Age Group (for filtering)
  age_group TEXT NOT NULL,                -- 'child', 'teen', 'young-adult', 'adult', 'senior'
  
  -- Persona Details
  bio TEXT,                               -- Short description
  learning_style TEXT,                    -- 'analytical', 'creative', 'social', etc.
  avatar_url TEXT,                        -- Path to avatar image
  
  -- Metadata
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for lookups
CREATE INDEX IF NOT EXISTS idx_personas_age_group ON learner_personas(age_group);
CREATE INDEX IF NOT EXISTS idx_personas_country ON learner_personas(country_code);

-- ═══════════════════════════════════════════════════════════════════
-- ENHANCED LESSON COMMENTS TABLE (v2)
-- Links to persona for richer display
-- ═══════════════════════════════════════════════════════════════════

-- Add persona_id column if not exists
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'lesson_comments' AND column_name = 'persona_id'
  ) THEN
    ALTER TABLE lesson_comments ADD COLUMN persona_id TEXT REFERENCES learner_personas(id);
  END IF;
END $$;

-- Add mood column for emotional context
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'lesson_comments' AND column_name = 'mood'
  ) THEN
    ALTER TABLE lesson_comments ADD COLUMN mood TEXT;
  END IF;
END $$;

-- Index for persona joins
CREATE INDEX IF NOT EXISTS idx_lesson_comments_persona ON lesson_comments(persona_id);

-- ═══════════════════════════════════════════════════════════════════
-- INSERT 60 DIVERSE PERSONAS
-- ═══════════════════════════════════════════════════════════════════

INSERT INTO learner_personas (id, name, age, country_code, country_flag, age_group, bio, learning_style, avatar_url) VALUES
-- North America
('emma-us', 'Emma', 28, 'US', '🇺🇸', 'adult', 'Software engineer, learns during lunch breaks', 'analytical', '/images/learners/emma-us.jpg'),
('marcus-us', 'Marcus', 16, 'US', '🇺🇸', 'teen', 'High school student, basketball player', 'curious', '/images/learners/marcus-us.jpg'),
('sarah-ca', 'Sarah', 45, 'CA', '🇨🇦', 'adult', 'Teacher, learns with her students', 'supportive', '/images/learners/sarah-ca.jpg'),
('joe-us', 'Joe', 72, 'US', '🇺🇸', 'senior', 'Retired engineer, lifelong learner', 'wise', '/images/learners/joe-us.jpg'),
('maya-mx', 'Maya', 34, 'MX', '🇲🇽', 'adult', 'Architect, loves connecting ideas', 'creative', '/images/learners/maya-mx.jpg'),

-- Europe
('james-uk', 'James', 31, 'GB', '🇬🇧', 'adult', 'Data analyst, morning learner', 'methodical', '/images/learners/james-uk.jpg'),
('charlotte-uk', 'Charlotte', 8, 'GB', '🇬🇧', 'child', 'Loves dinosaurs and space', 'wonder', '/images/learners/charlotte-uk.jpg'),
('marie-fr', 'Marie', 52, 'FR', '🇫🇷', 'adult', 'Museum curator, art lover', 'reflective', '/images/learners/marie-fr.jpg'),
('lucas-fr', 'Lucas', 19, 'FR', '🇫🇷', 'young-adult', 'University student, philosophy major', 'questioning', '/images/learners/lucas-fr.jpg'),
('hans-de', 'Hans', 67, 'DE', '🇩🇪', 'senior', 'Retired professor, still curious', 'scholarly', '/images/learners/hans-de.jpg'),
('lena-de', 'Lena', 24, 'DE', '🇩🇪', 'young-adult', 'Medical student, studies at night', 'dedicated', '/images/learners/lena-de.jpg'),
('isabella-it', 'Isabella', 38, 'IT', '🇮🇹', 'adult', 'Chef, connects food to history', 'passionate', '/images/learners/isabella-it.jpg'),
('sven-se', 'Sven', 29, 'SE', '🇸🇪', 'adult', 'Product designer, visual learner', 'creative', '/images/learners/sven-se.jpg'),
('nina-no', 'Nina', 41, 'NO', '🇳🇴', 'adult', 'Marine biologist, nature lover', 'scientific', '/images/learners/nina-no.jpg'),
('olga-ua', 'Olga', 33, 'UA', '🇺🇦', 'adult', 'Programmer, learns while commuting', 'efficient', '/images/learners/olga-ua.jpg'),

-- Asia
('yuki-jp', 'Yuki', 26, 'JP', '🇯🇵', 'young-adult', 'Graphic designer, anime fan', 'artistic', '/images/learners/yuki-jp.jpg'),
('haruto-jp', 'Haruto', 12, 'JP', '🇯🇵', 'child', 'Middle schooler, loves science', 'curious', '/images/learners/haruto-jp.jpg'),
('sakura-jp', 'Sakura', 58, 'JP', '🇯🇵', 'adult', 'Tea ceremony teacher, mindful learner', 'contemplative', '/images/learners/sakura-jp.jpg'),
('priya-in', 'Priya', 22, 'IN', '🇮🇳', 'young-adult', 'Engineering student, ambitious', 'driven', '/images/learners/priya-in.jpg'),
('arjun-in', 'Arjun', 35, 'IN', '🇮🇳', 'adult', 'Doctor, learns with his children', 'nurturing', '/images/learners/arjun-in.jpg'),
('ananya-in', 'Ananya', 9, 'IN', '🇮🇳', 'child', 'Loves drawing and stories', 'imaginative', '/images/learners/ananya-in.jpg'),
('wei-cn', 'Wei', 44, 'CN', '🇨🇳', 'adult', 'Business owner, practical learner', 'pragmatic', '/images/learners/wei-cn.jpg'),
('mei-cn', 'Mei', 17, 'CN', '🇨🇳', 'teen', 'High school senior, exam prep', 'focused', '/images/learners/mei-cn.jpg'),
('jin-kr', 'Jin', 27, 'KR', '🇰🇷', 'young-adult', 'Game developer, night owl', 'creative', '/images/learners/jin-kr.jpg'),
('soo-yeon-kr', 'Soo-yeon', 63, 'KR', '🇰🇷', 'senior', 'Grandmother, learning with grandkids', 'patient', '/images/learners/soo-yeon-kr.jpg'),

-- Middle East
('ahmed-eg', 'Ahmed', 30, 'EG', '🇪🇬', 'adult', 'History teacher, ancient cultures', 'storyteller', '/images/learners/ahmed-eg.jpg'),
('fatima-eg', 'Fatima', 21, 'EG', '🇪🇬', 'young-adult', 'Journalism student, curious', 'investigative', '/images/learners/fatima-eg.jpg'),
('omar-ae', 'Omar', 39, 'AE', '🇦🇪', 'adult', 'Entrepreneur, busy schedule', 'efficient', '/images/learners/omar-ae.jpg'),
('layla-ae', 'Layla', 14, 'AE', '🇦🇪', 'teen', 'Aspiring scientist, robotics club', 'experimental', '/images/learners/layla-ae.jpg'),

-- Africa
('kofi-gh', 'Kofi', 25, 'GH', '🇬🇭', 'young-adult', 'Agricultural engineer, community builder', 'practical', '/images/learners/kofi-gh.jpg'),
('ama-gh', 'Ama', 48, 'GH', '🇬🇭', 'adult', 'School principal, lifelong educator', 'mentoring', '/images/learners/ama-gh.jpg'),
('aisha-ke', 'Aisha', 20, 'KE', '🇰🇪', 'young-adult', 'Wildlife conservation student', 'passionate', '/images/learners/aisha-ke.jpg'),
('thabo-za', 'Thabo', 36, 'ZA', '🇿🇦', 'adult', 'Jazz musician, creative thinker', 'artistic', '/images/learners/thabo-za.jpg'),
('naledi-za', 'Naledi', 11, 'ZA', '🇿🇦', 'child', 'Loves math puzzles and soccer', 'playful', '/images/learners/naledi-za.jpg'),
('adebayo-ng', 'Adebayo', 42, 'NG', '🇳🇬', 'adult', 'Tech entrepreneur, Lagos', 'innovative', '/images/learners/adebayo-ng.jpg'),

-- South America
('maria-br', 'Maria', 28, 'BR', '🇧🇷', 'adult', 'Nurse, learns between shifts', 'compassionate', '/images/learners/maria-br.jpg'),
('pedro-br', 'Pedro', 55, 'BR', '🇧🇷', 'adult', 'Fisherman, loves ocean science', 'experiential', '/images/learners/pedro-br.jpg'),
('carlos-ar', 'Carlos', 32, 'AR', '🇦🇷', 'adult', 'Psychologist, interested in behavior', 'analytical', '/images/learners/carlos-ar.jpg'),
('diego-cl', 'Diego', 18, 'CL', '🇨🇱', 'young-adult', 'Astronomy enthusiast, stargazer', 'wonder', '/images/learners/diego-cl.jpg'),
('valentina-co', 'Valentina', 7, 'CO', '🇨🇴', 'child', 'Loves animals and colors', 'playful', '/images/learners/valentina-co.jpg'),

-- Oceania & SE Asia
('lisa-au', 'Lisa', 37, 'AU', '🇦🇺', 'adult', 'Environmental scientist, beach walks', 'observant', '/images/learners/lisa-au.jpg'),
('jack-nz', 'Jack', 23, 'NZ', '🇳🇿', 'young-adult', 'Outdoor guide, nature lover', 'adventurous', '/images/learners/jack-nz.jpg'),
('linh-vn', 'Linh', 29, 'VN', '🇻🇳', 'adult', 'Coffee shop owner, morning routine', 'steady', '/images/learners/linh-vn.jpg'),
('ling-sg', 'Ling', 45, 'SG', '🇸🇬', 'adult', 'Finance executive, efficient learner', 'structured', '/images/learners/ling-sg.jpg'),
('kai-th', 'Kai', 19, 'TH', '🇹🇭', 'young-adult', 'University student, travel lover', 'open-minded', '/images/learners/kai-th.jpg'),
('putri-id', 'Putri', 31, 'ID', '🇮🇩', 'adult', 'Teacher, passionate about education', 'nurturing', '/images/learners/putri-id.jpg'),

-- Additional Diversity
('zara-pk', 'Zara', 26, 'PK', '🇵🇰', 'young-adult', 'Social worker, community focus', 'empathetic', '/images/learners/zara-pk.jpg'),
('elena-ru', 'Elena', 40, 'RU', '🇷🇺', 'adult', 'Ballet teacher, disciplined', 'precise', '/images/learners/elena-ru.jpg'),
('tomasz-pl', 'Tomasz', 50, 'PL', '🇵🇱', 'adult', 'Carpenter, hands-on learner', 'practical', '/images/learners/tomasz-pl.jpg'),
('anna-gr', 'Anna', 65, 'GR', '🇬🇷', 'senior', 'Retired teacher, mythology lover', 'storytelling', '/images/learners/anna-gr.jpg'),
('chen-tw', 'Chen', 34, 'TW', '🇹🇼', 'adult', 'Chip designer, tech enthusiast', 'technical', '/images/learners/chen-tw.jpg'),
('fatou-sn', 'Fatou', 22, 'SN', '🇸🇳', 'young-adult', 'Medical student, community health', 'dedicated', '/images/learners/fatou-sn.jpg'),
('miguel-es', 'Miguel', 47, 'ES', '🇪🇸', 'adult', 'Chef, culinary arts lover', 'sensory', '/images/learners/miguel-es.jpg'),
('ana-pt', 'Ana', 15, 'PT', '🇵🇹', 'teen', 'Surfer, ocean science fan', 'active', '/images/learners/ana-pt.jpg')

ON CONFLICT (id) DO UPDATE SET
  name = EXCLUDED.name,
  age = EXCLUDED.age,
  bio = EXCLUDED.bio,
  learning_style = EXCLUDED.learning_style,
  avatar_url = EXCLUDED.avatar_url;

-- ═══════════════════════════════════════════════════════════════════
-- HELPER FUNCTION: Get random personas
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION get_random_personas(
  p_count INT DEFAULT 5,
  p_age_group TEXT DEFAULT NULL
)
RETURNS SETOF learner_personas
LANGUAGE sql
STABLE
AS $$
  SELECT * FROM learner_personas
  WHERE (p_age_group IS NULL OR age_group = p_age_group)
  ORDER BY random()
  LIMIT p_count;
$$;

-- ═══════════════════════════════════════════════════════════════════
-- VIEW: Comments with full persona data
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW lesson_comments_enriched AS
SELECT 
  lc.id,
  lc.lesson_day,
  lc.phase,
  lc.option_context,
  lc.comment_text,
  lc.comment_type,
  lc.mood,
  COALESCE(lp.name, lc.persona_name) as persona_name,
  COALESCE(lp.country_flag, lc.persona_flag) as persona_flag,
  COALESCE(lp.country_code, lc.persona_country) as persona_country,
  lp.avatar_url,
  lp.age,
  lp.age_group,
  lp.bio,
  lp.learning_style
FROM lesson_comments lc
LEFT JOIN learner_personas lp ON lc.persona_id = lp.id OR 
  (lc.persona_name = lp.name AND lc.persona_country = lp.country_code);

COMMENT ON VIEW lesson_comments_enriched IS 'Comments with full persona details for rich UI display';

-- ═══════════════════════════════════════════════════════════════════
-- RLS POLICIES
-- ═══════════════════════════════════════════════════════════════════

ALTER TABLE learner_personas ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read personas"
ON learner_personas FOR SELECT
TO authenticated, anon
USING (true);

CREATE POLICY "Service role manages personas"
ON learner_personas FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

