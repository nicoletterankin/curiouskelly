-- ═══════════════════════════════════════════════════════════════════════════
-- MIGRATION 004b: STORAGE BUCKET FOR LESSON ASSETS
-- ═══════════════════════════════════════════════════════════════════════════
-- 
-- Purpose: Create storage bucket for Kelly videos and other lesson assets
-- This bucket stores ElevenLabs Omnihuman generated videos and related media
--
-- Created: December 3, 2025
-- ═══════════════════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 1: Create storage bucket
-- ═══════════════════════════════════════════════════════════════════════════

-- Create the lesson-assets bucket (public for CDN access)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'lesson-assets',
  'lesson-assets',
  true,  -- Public bucket for fast CDN delivery
  104857600,  -- 100MB max file size (videos can be large)
  ARRAY[
    'video/mp4',
    'video/webm',
    'video/quicktime',
    'audio/mpeg',
    'audio/mp3',
    'audio/wav',
    'audio/ogg',
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/gif',
    'application/json'
  ]
)
ON CONFLICT (id) DO UPDATE SET
  public = EXCLUDED.public,
  file_size_limit = EXCLUDED.file_size_limit,
  allowed_mime_types = EXCLUDED.allowed_mime_types;

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 2: Storage policies for public read access
-- ═══════════════════════════════════════════════════════════════════════════

-- Drop existing policies if they exist (to allow re-running migration)
DROP POLICY IF EXISTS "Public read access for lesson assets" ON storage.objects;
DROP POLICY IF EXISTS "Authenticated read access for lesson assets" ON storage.objects;
DROP POLICY IF EXISTS "Service role full access for lesson assets" ON storage.objects;
DROP POLICY IF EXISTS "Service role insert access for lesson assets" ON storage.objects;
DROP POLICY IF EXISTS "Service role update access for lesson assets" ON storage.objects;
DROP POLICY IF EXISTS "Service role delete access for lesson assets" ON storage.objects;

-- Allow public read access (for CDN delivery)
CREATE POLICY "Public read access for lesson assets"
ON storage.objects FOR SELECT
TO anon
USING (bucket_id = 'lesson-assets');

-- Allow authenticated users to read
CREATE POLICY "Authenticated read access for lesson assets"
ON storage.objects FOR SELECT
TO authenticated
USING (bucket_id = 'lesson-assets');

-- Allow service role to insert (for video generation)
CREATE POLICY "Service role insert access for lesson assets"
ON storage.objects FOR INSERT
TO service_role
WITH CHECK (bucket_id = 'lesson-assets');

-- Allow service role to update (for replacing videos)
CREATE POLICY "Service role update access for lesson assets"
ON storage.objects FOR UPDATE
TO service_role
USING (bucket_id = 'lesson-assets');

-- Allow service role to delete (for cleanup)
CREATE POLICY "Service role delete access for lesson assets"
ON storage.objects FOR DELETE
TO service_role
USING (bucket_id = 'lesson-assets');

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 3: Expected folder structure in bucket
-- ═══════════════════════════════════════════════════════════════════════════
--
-- lesson-assets/
-- ├── kelly-videos/                    # ElevenLabs Omnihuman videos
-- │   ├── 1/                           # Day 1
-- │   │   ├── welcome/
-- │   │   │   ├── young_adult-en.mp4   # Age bucket + language variant
-- │   │   │   ├── child-en.mp4
-- │   │   │   └── elder-es.mp4
-- │   │   ├── q1/
-- │   │   ├── q2/
-- │   │   ├── q3/
-- │   │   └── wisdom/
-- │   ├── 2/
-- │   └── .../
-- │
-- ├── kelly-audio/                     # Pre-generated TTS audio
-- │   ├── 1/
-- │   │   ├── welcome/
-- │   │   │   └── young_adult-en.mp3
-- │   │   └── .../
-- │   └── .../
-- │
-- ├── thumbnails/                      # Lesson thumbnails
-- │   └── raw/
-- │       ├── lesson-001-topic.png
-- │       └── .../
-- │
-- └── expressions/                     # Pre-computed expression data
--     ├── 1/
--     │   └── welcome-young_adult-en.json
--     └── .../

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 4: Helper function to construct video path
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION get_kelly_video_storage_path(
  p_lesson_day INTEGER,
  p_phase TEXT,
  p_age_bucket TEXT,
  p_language TEXT DEFAULT 'en'
)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN 'kelly-videos/' || p_lesson_day || '/' || p_phase || '/' || p_age_bucket || '-' || p_language || '.mp4';
END;
$$;

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 5: Helper function to construct audio path
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION get_kelly_audio_storage_path(
  p_lesson_day INTEGER,
  p_phase TEXT,
  p_age_bucket TEXT,
  p_language TEXT DEFAULT 'en'
)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN 'kelly-audio/' || p_lesson_day || '/' || p_phase || '/' || p_age_bucket || '-' || p_language || '.mp3';
END;
$$;

-- ═══════════════════════════════════════════════════════════════════════════
-- VERIFICATION
-- ═══════════════════════════════════════════════════════════════════════════

SELECT 'Migration 004b: Storage bucket setup complete!' as status;

-- Verify bucket exists
SELECT 
  'lesson-assets bucket' as check_name,
  EXISTS (
    SELECT 1 FROM storage.buckets WHERE id = 'lesson-assets'
  ) as exists;

-- List current policies
SELECT 
  policyname as policy_name,
  tablename as table_name,
  roles,
  cmd as operation
FROM pg_policies 
WHERE tablename = 'objects' 
  AND schemaname = 'storage'
  AND policyname LIKE '%lesson assets%';



