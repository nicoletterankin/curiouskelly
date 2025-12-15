-- ============================================
-- COMMUNITY FEATURES: Comments & Artwork
-- ============================================

-- ============================================
-- LESSON COMMENTS
-- ============================================

CREATE TABLE IF NOT EXISTS public.lesson_comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Author and lesson
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  day_number INTEGER NOT NULL,
  
  -- Content
  content TEXT NOT NULL CHECK (char_length(content) <= 2000),
  content_html TEXT,  -- Sanitized HTML for display
  
  -- Threading (flat for now, parent_comment_id for future)
  parent_comment_id UUID REFERENCES public.lesson_comments(id) ON DELETE CASCADE,
  thread_depth INTEGER DEFAULT 0 CHECK (thread_depth <= 1),  -- Max 1 level of replies
  
  -- Moderation
  status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'flagged')),
  moderated_by UUID REFERENCES public.users(id),
  moderated_at TIMESTAMPTZ,
  rejection_reason TEXT,
  ai_moderation_score DECIMAL(5,4),  -- 0-1 confidence of appropriateness
  
  -- Engagement
  upvotes INTEGER DEFAULT 0,
  reports INTEGER DEFAULT 0,
  
  -- Kelly response
  kelly_responded BOOLEAN DEFAULT false,
  kelly_response_id UUID REFERENCES public.lesson_comments(id),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  edited_at TIMESTAMPTZ,
  
  -- Soft delete
  deleted_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_comments_day ON public.lesson_comments(day_number) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_comments_user ON public.lesson_comments(user_id);
CREATE INDEX IF NOT EXISTS idx_comments_status ON public.lesson_comments(status);
CREATE INDEX IF NOT EXISTS idx_comments_pending ON public.lesson_comments(created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_comments_approved ON public.lesson_comments(day_number, created_at DESC) WHERE status = 'approved' AND deleted_at IS NULL;

-- Enable RLS
ALTER TABLE public.lesson_comments ENABLE ROW LEVEL SECURITY;

-- Users can view approved comments
CREATE POLICY "Anyone can view approved comments" ON public.lesson_comments
  FOR SELECT USING (status = 'approved' AND deleted_at IS NULL);

-- Users can view their own comments (any status)
CREATE POLICY "Users can view own comments" ON public.lesson_comments
  FOR SELECT USING (auth.uid() = user_id);

-- Users can post comments
CREATE POLICY "Users can post comments" ON public.lesson_comments
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can edit their own comments
CREATE POLICY "Users can edit own comments" ON public.lesson_comments
  FOR UPDATE USING (auth.uid() = user_id AND deleted_at IS NULL);

-- Users can soft-delete their own comments
CREATE POLICY "Users can delete own comments" ON public.lesson_comments
  FOR UPDATE USING (auth.uid() = user_id)
  WITH CHECK (deleted_at IS NOT NULL);

-- Trigger to update user's lifetime_contributions
CREATE OR REPLACE FUNCTION update_user_contributions_on_comment()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' AND NEW.status = 'approved' THEN
    UPDATE public.users SET lifetime_contributions = lifetime_contributions + 1 WHERE id = NEW.user_id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS track_comment_contribution ON public.lesson_comments;

CREATE TRIGGER track_comment_contribution
  AFTER INSERT OR UPDATE ON public.lesson_comments
  FOR EACH ROW
  WHEN (NEW.status = 'approved')
  EXECUTE FUNCTION update_user_contributions_on_comment();

-- Comment
COMMENT ON TABLE public.lesson_comments IS 'User comments on lessons with moderation';

-- ============================================
-- LESSON ARTWORK SUBMISSIONS
-- ============================================

CREATE TABLE IF NOT EXISTS public.lesson_artwork_submissions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Submitter and lesson
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  day_number INTEGER NOT NULL,
  
  -- Asset
  image_url TEXT NOT NULL,
  thumbnail_url TEXT,
  storage_path TEXT,  -- Path in Supabase Storage
  original_filename TEXT,
  file_size_bytes INTEGER,
  dimensions VARCHAR(20),  -- '1920x1080'
  
  -- Metadata
  title VARCHAR(200),
  description TEXT CHECK (char_length(description) <= 1000),
  ai_generated BOOLEAN DEFAULT FALSE,
  tools_used TEXT[],  -- ['photoshop', 'midjourney', etc.]
  
  -- Moderation
  status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'featured')),
  moderated_by UUID REFERENCES public.users(id),
  moderated_at TIMESTAMPTZ,
  rejection_reason TEXT,
  
  -- Usage tracking
  times_displayed INTEGER DEFAULT 0,
  selected_as_official BOOLEAN DEFAULT FALSE,
  
  -- License agreement
  license_agreed BOOLEAN DEFAULT TRUE,
  license_agreed_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Withdrawn
  withdrawn_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_artwork_day ON public.lesson_artwork_submissions(day_number) WHERE withdrawn_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_artwork_user ON public.lesson_artwork_submissions(user_id);
CREATE INDEX IF NOT EXISTS idx_artwork_status ON public.lesson_artwork_submissions(status);
CREATE INDEX IF NOT EXISTS idx_artwork_approved ON public.lesson_artwork_submissions(day_number) WHERE status IN ('approved', 'featured') AND withdrawn_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_artwork_featured ON public.lesson_artwork_submissions(day_number) WHERE status = 'featured' AND withdrawn_at IS NULL;

-- Enable RLS
ALTER TABLE public.lesson_artwork_submissions ENABLE ROW LEVEL SECURITY;

-- Users can view approved artwork
CREATE POLICY "Anyone can view approved artwork" ON public.lesson_artwork_submissions
  FOR SELECT USING (status IN ('approved', 'featured') AND withdrawn_at IS NULL);

-- Users can view their own submissions
CREATE POLICY "Users can view own artwork" ON public.lesson_artwork_submissions
  FOR SELECT USING (auth.uid() = user_id);

-- Users can submit artwork
CREATE POLICY "Users can submit artwork" ON public.lesson_artwork_submissions
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can withdraw their own artwork
CREATE POLICY "Users can withdraw artwork" ON public.lesson_artwork_submissions
  FOR UPDATE USING (auth.uid() = user_id);

-- Trigger to update user's lifetime_contributions
CREATE OR REPLACE FUNCTION update_user_contributions_on_artwork()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND NEW.status = 'approved' AND OLD.status != 'approved') THEN
    UPDATE public.users SET lifetime_contributions = lifetime_contributions + 1 WHERE id = NEW.user_id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS track_artwork_contribution ON public.lesson_artwork_submissions;

CREATE TRIGGER track_artwork_contribution
  AFTER INSERT OR UPDATE ON public.lesson_artwork_submissions
  FOR EACH ROW
  WHEN (NEW.status IN ('approved', 'featured'))
  EXECUTE FUNCTION update_user_contributions_on_artwork();

-- Comment
COMMENT ON TABLE public.lesson_artwork_submissions IS 'User-submitted artwork for lessons';

-- ============================================
-- COMMENT REACTIONS (upvotes)
-- ============================================

CREATE TABLE IF NOT EXISTS public.comment_reactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  comment_id UUID NOT NULL REFERENCES public.lesson_comments(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  reaction_type VARCHAR(20) DEFAULT 'upvote' CHECK (reaction_type IN ('upvote', 'heart', 'laugh', 'think')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(comment_id, user_id, reaction_type)
);

-- Index
CREATE INDEX IF NOT EXISTS idx_reactions_comment ON public.comment_reactions(comment_id);
CREATE INDEX IF NOT EXISTS idx_reactions_user ON public.comment_reactions(user_id);

-- Enable RLS
ALTER TABLE public.comment_reactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view reactions" ON public.comment_reactions FOR SELECT USING (true);
CREATE POLICY "Users can add reactions" ON public.comment_reactions FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can remove own reactions" ON public.comment_reactions FOR DELETE USING (auth.uid() = user_id);

-- Trigger to update comment upvote count
CREATE OR REPLACE FUNCTION update_comment_upvotes()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' AND NEW.reaction_type = 'upvote' THEN
    UPDATE public.lesson_comments SET upvotes = upvotes + 1 WHERE id = NEW.comment_id;
  ELSIF TG_OP = 'DELETE' AND OLD.reaction_type = 'upvote' THEN
    UPDATE public.lesson_comments SET upvotes = upvotes - 1 WHERE id = OLD.comment_id;
  END IF;
  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS sync_comment_upvotes ON public.comment_reactions;

CREATE TRIGGER sync_comment_upvotes
  AFTER INSERT OR DELETE ON public.comment_reactions
  FOR EACH ROW EXECUTE FUNCTION update_comment_upvotes();
