-- Phase Comments and Curriculum Suggestions Migration
-- Created: 2025-12-17
-- Purpose: Enable per-phase learner comments and curriculum improvement suggestions

-- =============================================================================
-- PHASE COMMENTS TABLE
-- =============================================================================
-- Learners can leave comments on specific phases of lessons
-- Comments can be insights, questions, suggestions, or experiences

CREATE TABLE IF NOT EXISTS phase_comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  
  -- Lesson identification
  lesson_day INTEGER NOT NULL CHECK (lesson_day >= 1 AND lesson_day <= 365),
  lesson_year INTEGER NOT NULL DEFAULT 1 CHECK (lesson_year IN (1, 2)), -- 1=LEARN, 2=GROW
  phase TEXT NOT NULL CHECK (phase IN ('hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro')),
  
  -- Comment content
  comment_type TEXT NOT NULL CHECK (comment_type IN ('insight', 'question', 'suggestion', 'experience')),
  content TEXT NOT NULL CHECK (LENGTH(content) >= 10 AND LENGTH(content) <= 2000),
  
  -- Engagement metrics
  upvotes INTEGER DEFAULT 0 CHECK (upvotes >= 0),
  reply_count INTEGER DEFAULT 0 CHECK (reply_count >= 0),
  
  -- Moderation
  moderation_status TEXT DEFAULT 'pending' CHECK (moderation_status IN ('pending', 'approved', 'hidden', 'featured')),
  moderated_at TIMESTAMP WITH TIME ZONE,
  moderated_by UUID REFERENCES auth.users(id),
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  -- Soft delete
  deleted_at TIMESTAMP WITH TIME ZONE
);

-- =============================================================================
-- COMMENT REPLIES TABLE
-- =============================================================================
-- Threaded replies to comments

CREATE TABLE IF NOT EXISTS comment_replies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  comment_id UUID REFERENCES phase_comments(id) ON DELETE CASCADE NOT NULL,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  
  -- Reply content
  content TEXT NOT NULL CHECK (LENGTH(content) >= 1 AND LENGTH(content) <= 1000),
  
  -- Engagement
  upvotes INTEGER DEFAULT 0 CHECK (upvotes >= 0),
  
  -- Moderation
  moderation_status TEXT DEFAULT 'approved' CHECK (moderation_status IN ('pending', 'approved', 'hidden')),
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  deleted_at TIMESTAMP WITH TIME ZONE
);

-- =============================================================================
-- COMMENT VOTES TABLE
-- =============================================================================
-- Track upvotes/downvotes on comments (one vote per user per comment)

CREATE TABLE IF NOT EXISTS comment_votes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  comment_id UUID REFERENCES phase_comments(id) ON DELETE CASCADE NOT NULL,
  
  -- Vote type (only upvotes for now, extensible)
  vote_type TEXT NOT NULL CHECK (vote_type IN ('up', 'down')) DEFAULT 'up',
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  -- Ensure one vote per user per comment
  UNIQUE(user_id, comment_id)
);

-- =============================================================================
-- CURRICULUM SUGGESTIONS TABLE
-- =============================================================================
-- Structured suggestions for improving curriculum content

CREATE TABLE IF NOT EXISTS curriculum_suggestions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  
  -- Lesson identification
  lesson_day INTEGER NOT NULL CHECK (lesson_day >= 1 AND lesson_day <= 365),
  lesson_year INTEGER NOT NULL DEFAULT 1 CHECK (lesson_year IN (1, 2)),
  phase TEXT CHECK (phase IS NULL OR phase IN ('hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro')),
  
  -- Suggestion details
  suggestion_type TEXT NOT NULL CHECK (suggestion_type IN ('correction', 'addition', 'alternative', 'removal', 'clarification')),
  current_content TEXT, -- The current content being suggested for change
  suggested_content TEXT NOT NULL CHECK (LENGTH(suggested_content) >= 10),
  rationale TEXT CHECK (rationale IS NULL OR LENGTH(rationale) <= 2000),
  
  -- Review status
  status TEXT DEFAULT 'submitted' CHECK (status IN ('submitted', 'under_review', 'accepted', 'declined', 'implemented')),
  reviewed_at TIMESTAMP WITH TIME ZONE,
  reviewed_by UUID REFERENCES auth.users(id),
  review_notes TEXT,
  
  -- Implementation tracking
  implemented_at TIMESTAMP WITH TIME ZONE,
  implementation_version TEXT, -- Version where change was applied
  
  -- Attribution (if accepted)
  credited_publicly BOOLEAN DEFAULT true,
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Phase comments indexes
CREATE INDEX idx_phase_comments_lesson ON phase_comments(lesson_year, lesson_day, phase) WHERE deleted_at IS NULL;
CREATE INDEX idx_phase_comments_user ON phase_comments(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_phase_comments_status ON phase_comments(moderation_status) WHERE deleted_at IS NULL;
CREATE INDEX idx_phase_comments_created ON phase_comments(created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX idx_phase_comments_upvotes ON phase_comments(upvotes DESC) WHERE deleted_at IS NULL AND moderation_status = 'approved';

-- Comment replies indexes
CREATE INDEX idx_comment_replies_comment ON comment_replies(comment_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_comment_replies_user ON comment_replies(user_id) WHERE deleted_at IS NULL;

-- Comment votes indexes
CREATE INDEX idx_comment_votes_comment ON comment_votes(comment_id);
CREATE INDEX idx_comment_votes_user ON comment_votes(user_id);

-- Curriculum suggestions indexes
CREATE INDEX idx_curriculum_suggestions_lesson ON curriculum_suggestions(lesson_year, lesson_day);
CREATE INDEX idx_curriculum_suggestions_user ON curriculum_suggestions(user_id);
CREATE INDEX idx_curriculum_suggestions_status ON curriculum_suggestions(status);
CREATE INDEX idx_curriculum_suggestions_pending ON curriculum_suggestions(created_at) WHERE status = 'submitted';

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_phase_comments_updated_at
  BEFORE UPDATE ON phase_comments
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_curriculum_suggestions_updated_at
  BEFORE UPDATE ON curriculum_suggestions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Auto-update upvote counts
CREATE OR REPLACE FUNCTION update_comment_upvotes()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE phase_comments 
    SET upvotes = upvotes + CASE WHEN NEW.vote_type = 'up' THEN 1 ELSE 0 END
    WHERE id = NEW.comment_id;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE phase_comments 
    SET upvotes = upvotes - CASE WHEN OLD.vote_type = 'up' THEN 1 ELSE 0 END
    WHERE id = OLD.comment_id;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_comment_upvotes
  AFTER INSERT OR DELETE ON comment_votes
  FOR EACH ROW
  EXECUTE FUNCTION update_comment_upvotes();

-- Auto-update reply counts
CREATE OR REPLACE FUNCTION update_reply_count()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE phase_comments 
    SET reply_count = reply_count + 1
    WHERE id = NEW.comment_id;
  ELSIF TG_OP = 'DELETE' OR (TG_OP = 'UPDATE' AND NEW.deleted_at IS NOT NULL AND OLD.deleted_at IS NULL) THEN
    UPDATE phase_comments 
    SET reply_count = reply_count - 1
    WHERE id = COALESCE(NEW.comment_id, OLD.comment_id);
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_reply_count
  AFTER INSERT OR UPDATE OR DELETE ON comment_replies
  FOR EACH ROW
  EXECUTE FUNCTION update_reply_count();

-- =============================================================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE phase_comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE comment_replies ENABLE ROW LEVEL SECURITY;
ALTER TABLE comment_votes ENABLE ROW LEVEL SECURITY;
ALTER TABLE curriculum_suggestions ENABLE ROW LEVEL SECURITY;

-- Phase comments policies
CREATE POLICY "Anyone can view approved comments"
  ON phase_comments FOR SELECT
  USING (moderation_status IN ('approved', 'featured') AND deleted_at IS NULL);

CREATE POLICY "Users can view their own comments"
  ON phase_comments FOR SELECT
  USING (auth.uid() = user_id AND deleted_at IS NULL);

CREATE POLICY "Authenticated users can create comments"
  ON phase_comments FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own comments"
  ON phase_comments FOR UPDATE
  USING (auth.uid() = user_id AND deleted_at IS NULL)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can soft delete their own comments"
  ON phase_comments FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id AND deleted_at IS NOT NULL);

-- Comment replies policies
CREATE POLICY "Anyone can view approved replies"
  ON comment_replies FOR SELECT
  USING (moderation_status = 'approved' AND deleted_at IS NULL);

CREATE POLICY "Users can view their own replies"
  ON comment_replies FOR SELECT
  USING (auth.uid() = user_id AND deleted_at IS NULL);

CREATE POLICY "Authenticated users can create replies"
  ON comment_replies FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own replies"
  ON comment_replies FOR UPDATE
  USING (auth.uid() = user_id AND deleted_at IS NULL);

-- Comment votes policies
CREATE POLICY "Anyone can view votes"
  ON comment_votes FOR SELECT
  USING (true);

CREATE POLICY "Authenticated users can vote"
  ON comment_votes FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can remove their own votes"
  ON comment_votes FOR DELETE
  USING (auth.uid() = user_id);

-- Curriculum suggestions policies
CREATE POLICY "Users can view their own suggestions"
  ON curriculum_suggestions FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Anyone can view implemented suggestions"
  ON curriculum_suggestions FOR SELECT
  USING (status = 'implemented' AND credited_publicly = true);

CREATE POLICY "Authenticated users can create suggestions"
  ON curriculum_suggestions FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their pending suggestions"
  ON curriculum_suggestions FOR UPDATE
  USING (auth.uid() = user_id AND status = 'submitted')
  WITH CHECK (auth.uid() = user_id);

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- View for getting comment counts per phase
CREATE OR REPLACE VIEW phase_comment_counts AS
SELECT 
  lesson_year,
  lesson_day,
  phase,
  COUNT(*) FILTER (WHERE moderation_status = 'approved') AS approved_count,
  COUNT(*) FILTER (WHERE moderation_status = 'featured') AS featured_count,
  COUNT(*) AS total_count
FROM phase_comments
WHERE deleted_at IS NULL
GROUP BY lesson_year, lesson_day, phase;

-- View for top comments per lesson
CREATE OR REPLACE VIEW top_phase_comments AS
SELECT 
  pc.*,
  u.email as user_email
FROM phase_comments pc
LEFT JOIN auth.users u ON pc.user_id = u.id
WHERE pc.moderation_status IN ('approved', 'featured')
  AND pc.deleted_at IS NULL
ORDER BY 
  pc.moderation_status = 'featured' DESC,
  pc.upvotes DESC,
  pc.created_at DESC;

-- =============================================================================
-- SAMPLE DATA (OPTIONAL - COMMENT OUT IN PRODUCTION)
-- =============================================================================

-- Uncomment to add sample data for testing:
/*
INSERT INTO phase_comments (user_id, lesson_day, lesson_year, phase, comment_type, content, moderation_status)
VALUES 
  ('00000000-0000-0000-0000-000000000000', 1, 1, 'hook', 'insight', 'This opening really grabbed my attention! The question about fresh starts resonated with me.', 'approved'),
  ('00000000-0000-0000-0000-000000000000', 1, 1, 'fact1', 'question', 'Is the 66-day habit formation finding consistent across all types of habits?', 'approved');
*/

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

-- Grant access to authenticated users
GRANT SELECT, INSERT, UPDATE ON phase_comments TO authenticated;
GRANT SELECT, INSERT, UPDATE ON comment_replies TO authenticated;
GRANT SELECT, INSERT, DELETE ON comment_votes TO authenticated;
GRANT SELECT, INSERT, UPDATE ON curriculum_suggestions TO authenticated;

-- Grant access to views
GRANT SELECT ON phase_comment_counts TO authenticated;
GRANT SELECT ON top_phase_comments TO authenticated;

-- Service role gets full access
GRANT ALL ON phase_comments TO service_role;
GRANT ALL ON comment_replies TO service_role;
GRANT ALL ON comment_votes TO service_role;
GRANT ALL ON curriculum_suggestions TO service_role;
