-- ============================================================================
-- TRANSLATION COMMONS SCHEMA
-- ============================================================================
-- Enables BYOK translation system where users contribute translations
-- using their own API keys, and all translations are saved to commons.

-- Translation Commons: All translated content
CREATE TABLE IF NOT EXISTS public.translation_commons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- What's being translated
  unit_id TEXT NOT NULL,              -- "day-1.phases.hook.script"
  source_lang TEXT NOT NULL DEFAULT 'en',
  target_lang TEXT NOT NULL,          -- "es", "pt", etc.
  
  -- The translation
  source_text TEXT NOT NULL,
  translated_text TEXT NOT NULL,
  
  -- Quality & provenance
  provider TEXT NOT NULL,             -- "openai", "google", "deepl", "anthropic"
  provider_model TEXT,                -- "gpt-4o-mini", "gemini-pro", etc.
  quality_scores JSONB DEFAULT '{}',  -- { length_ratio, voice_score, etc. }
  
  -- Attribution
  contributed_by UUID REFERENCES auth.users(id),
  contributed_by_display_name TEXT,
  contributed_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Lifecycle
  status TEXT DEFAULT 'draft',        -- draft, validated, flagged, approved, published, rejected
  reviewed_by UUID REFERENCES auth.users(id),
  reviewed_at TIMESTAMPTZ,
  rejection_reason TEXT,
  
  -- Versioning
  source_version TEXT,                -- Hash of source text at time of translation
  version INTEGER DEFAULT 1,
  is_current BOOLEAN DEFAULT TRUE,
  replaced_by UUID REFERENCES translation_commons(id),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_translation_commons_unit ON translation_commons(unit_id, target_lang);
CREATE INDEX IF NOT EXISTS idx_translation_commons_lang ON translation_commons(target_lang, status);
CREATE INDEX IF NOT EXISTS idx_translation_commons_contributor ON translation_commons(contributed_by);
CREATE INDEX IF NOT EXISTS idx_translation_commons_current ON translation_commons(unit_id, target_lang, is_current) WHERE is_current = TRUE;

-- Unique constraint for current version
CREATE UNIQUE INDEX IF NOT EXISTS idx_translation_commons_unique_current 
  ON translation_commons(unit_id, target_lang) 
  WHERE is_current = TRUE;

-- Enable RLS
ALTER TABLE translation_commons ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Anyone can read published translations"
  ON translation_commons FOR SELECT
  USING (status = 'published');

CREATE POLICY "Users can read own contributions"
  ON translation_commons FOR SELECT
  USING (contributed_by = auth.uid());

CREATE POLICY "Authenticated users can contribute"
  ON translation_commons FOR INSERT
  WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Contributors can update own pending"
  ON translation_commons FOR UPDATE
  USING (contributed_by = auth.uid() AND status IN ('draft', 'flagged'));


-- ============================================================================
-- LANGUAGE PROGRESS TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.translation_progress (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  target_lang TEXT NOT NULL,
  day_number INTEGER,                -- NULL = overall for language
  
  -- Progress counts
  total_units INTEGER DEFAULT 0,
  translated_count INTEGER DEFAULT 0,
  validated_count INTEGER DEFAULT 0,
  published_count INTEGER DEFAULT 0,
  
  -- Completion
  is_complete BOOLEAN DEFAULT FALSE,
  completed_at TIMESTAMPTZ,
  
  -- Timestamps
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(target_lang, day_number)
);

-- Enable RLS
ALTER TABLE translation_progress ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read progress"
  ON translation_progress FOR SELECT
  USING (true);


-- ============================================================================
-- USER TRANSLATION STATS
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.user_translation_stats (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id),
  
  -- Contribution counts
  translations_contributed INTEGER DEFAULT 0,
  translations_approved INTEGER DEFAULT 0,
  translations_rejected INTEGER DEFAULT 0,
  translations_pending INTEGER DEFAULT 0,
  
  -- Languages
  languages_contributed TEXT[] DEFAULT '{}',
  
  -- Recognition
  contribution_rank TEXT DEFAULT 'Newcomer',  -- Newcomer, Bronze, Silver, Gold, Platinum
  badges JSONB DEFAULT '[]',
  
  -- Activity
  first_contribution_at TIMESTAMPTZ,
  last_contribution_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE user_translation_stats ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own stats"
  ON user_translation_stats FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Anyone can read leaderboard stats"
  ON user_translation_stats FOR SELECT
  USING (contribution_rank != 'Newcomer');


-- ============================================================================
-- LANGUAGE SPONSORSHIPS
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.language_sponsorships (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Sponsor info
  user_id UUID REFERENCES auth.users(id),
  display_name TEXT,
  
  -- Language
  target_lang TEXT NOT NULL,
  
  -- Payment
  payment_type TEXT NOT NULL,         -- 'stripe', 'byok', 'hybrid'
  stripe_payment_id TEXT,
  amount_usd DECIMAL(10, 2),
  byok_provider TEXT,
  byok_credits_used INTEGER DEFAULT 0,
  
  -- Status
  status TEXT DEFAULT 'pending',      -- pending, active, completed, cancelled
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ
);

-- Enable RLS
ALTER TABLE language_sponsorships ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own sponsorships"
  ON language_sponsorships FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Anyone can see completed sponsorships"
  ON language_sponsorships FOR SELECT
  USING (status = 'completed');


-- ============================================================================
-- BYOK KEY REGISTRY (Encrypted)
-- ============================================================================
-- Note: Keys should be encrypted client-side before storage
-- This table only stores encrypted keys, never plaintext

CREATE TABLE IF NOT EXISTS public.byok_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id),
  
  -- Provider
  provider TEXT NOT NULL,             -- 'google', 'openai', 'anthropic', 'deepl'
  
  -- Encrypted key (client-side encrypted)
  encrypted_key TEXT NOT NULL,
  key_hint TEXT,                      -- Last 4 chars for identification
  
  -- Permissions
  allowed_purposes TEXT[] DEFAULT '{translation}',
  allowed_languages TEXT[] DEFAULT '{}',  -- Empty = all languages
  
  -- Usage tracking
  total_requests INTEGER DEFAULT 0,
  total_tokens INTEGER DEFAULT 0,
  last_used_at TIMESTAMPTZ,
  
  -- Status
  is_active BOOLEAN DEFAULT TRUE,
  is_verified BOOLEAN DEFAULT FALSE,
  verified_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE byok_keys ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own keys"
  ON byok_keys FOR ALL
  USING (user_id = auth.uid());


-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_translation_commons_updated_at
  BEFORE UPDATE ON translation_commons
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_translation_progress_updated_at
  BEFORE UPDATE ON translation_progress
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_translation_stats_updated_at
  BEFORE UPDATE ON user_translation_stats
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_byok_keys_updated_at
  BEFORE UPDATE ON byok_keys
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Get translation progress for a language
CREATE OR REPLACE FUNCTION get_language_progress(lang TEXT)
RETURNS TABLE (
  total_days INTEGER,
  completed_days INTEGER,
  total_units INTEGER,
  translated_units INTEGER,
  percent_complete DECIMAL
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    365 AS total_days,
    COUNT(DISTINCT CASE WHEN is_complete THEN day_number END)::INTEGER AS completed_days,
    SUM(total_units)::INTEGER AS total_units,
    SUM(published_count)::INTEGER AS translated_units,
    ROUND((SUM(published_count)::DECIMAL / NULLIF(SUM(total_units), 0)) * 100, 2) AS percent_complete
  FROM translation_progress
  WHERE target_lang = lang;
END;
$$ LANGUAGE plpgsql;

-- Get leaderboard
CREATE OR REPLACE FUNCTION get_translation_leaderboard(limit_count INTEGER DEFAULT 10)
RETURNS TABLE (
  user_id UUID,
  display_name TEXT,
  translations_approved INTEGER,
  languages TEXT[],
  rank TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    uts.user_id,
    COALESCE(u.raw_user_meta_data->>'display_name', 'Anonymous') AS display_name,
    uts.translations_approved,
    uts.languages_contributed AS languages,
    uts.contribution_rank AS rank
  FROM user_translation_stats uts
  LEFT JOIN auth.users u ON u.id = uts.user_id
  WHERE uts.translations_approved > 0
  ORDER BY uts.translations_approved DESC
  LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
