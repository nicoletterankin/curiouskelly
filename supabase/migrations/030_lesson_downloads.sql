-- ============================================
-- LESSON DOWNLOADS: Offline Access
-- ============================================

CREATE TABLE IF NOT EXISTS public.lesson_downloads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Who downloaded
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- What type of download
  download_type VARCHAR(20) NOT NULL CHECK (download_type IN ('single', 'week', 'month', 'bundle', 'update')),
  
  -- Which lessons
  day_numbers INTEGER[] NOT NULL,  -- Array of day numbers included
  lessons_count INTEGER NOT NULL,
  
  -- Bundle info
  bundle_version VARCHAR(20),
  file_size_bytes BIGINT,
  
  -- Download key (for encrypted access)
  download_key_hash VARCHAR(64),  -- SHA-256 of the key
  key_expires_at TIMESTAMPTZ,
  
  -- Status
  status VARCHAR(20) DEFAULT 'requested' CHECK (status IN ('requested', 'generating', 'ready', 'downloaded', 'expired', 'failed')),
  download_url TEXT,
  
  -- Timestamps
  requested_at TIMESTAMPTZ DEFAULT NOW(),
  ready_at TIMESTAMPTZ,
  downloaded_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ,  -- Downloads expire after X days
  
  -- Access verification
  subscription_verified_at TIMESTAMPTZ,  -- When we last verified they have access
  
  -- Device
  device_id VARCHAR(100),
  platform VARCHAR(20)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_downloads_user ON public.lesson_downloads(user_id);
CREATE INDEX IF NOT EXISTS idx_downloads_status ON public.lesson_downloads(status);
CREATE INDEX IF NOT EXISTS idx_downloads_requested ON public.lesson_downloads(requested_at DESC);

-- Enable RLS
ALTER TABLE public.lesson_downloads ENABLE ROW LEVEL SECURITY;

-- Users can view their own downloads
CREATE POLICY "Users can view own downloads" ON public.lesson_downloads
  FOR SELECT USING (auth.uid() = user_id);

-- Users can request downloads
CREATE POLICY "Users can request downloads" ON public.lesson_downloads
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can update their own downloads
CREATE POLICY "Users can update own downloads" ON public.lesson_downloads
  FOR UPDATE USING (auth.uid() = user_id);

-- ============================================
-- OFFLINE SYNC CHECKPOINTS
-- ============================================
-- Track what's been synced for offline users

CREATE TABLE IF NOT EXISTS public.offline_sync_checkpoints (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  device_id VARCHAR(100) NOT NULL,
  
  -- What's synced
  last_progress_sync TIMESTAMPTZ,
  last_events_sync TIMESTAMPTZ,
  pending_events_count INTEGER DEFAULT 0,
  
  -- Device info
  platform VARCHAR(20),
  app_version VARCHAR(20),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(user_id, device_id)
);

-- Index
CREATE INDEX IF NOT EXISTS idx_sync_user ON public.offline_sync_checkpoints(user_id);

-- Enable RLS
ALTER TABLE public.offline_sync_checkpoints ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own sync" ON public.offline_sync_checkpoints
  FOR ALL USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_sync_checkpoints_updated_at
  BEFORE UPDATE ON public.offline_sync_checkpoints
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE public.lesson_downloads IS 'Track lesson downloads for offline access';
COMMENT ON TABLE public.offline_sync_checkpoints IS 'Track sync state per device for offline users';
