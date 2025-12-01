-- Kelly Visual Identity Asset Management System
-- Migration: Create kelly_assets table and views
-- Date: 2025-11-30

-- Main asset management table
CREATE TABLE IF NOT EXISTS kelly_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- File reference
    filename VARCHAR(255) NOT NULL,
    r2_key TEXT NOT NULL,
    r2_bucket VARCHAR(100) DEFAULT 'kelly-assets',
    
    -- Character state
    pose_type VARCHAR(50) NOT NULL,
    pose_direction VARCHAR(20),
    emotion VARCHAR(50),
    
    -- Workflow
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'review', 'approved', 'published', 'archived')),
    is_hero BOOLEAN DEFAULT false,
    version INTEGER DEFAULT 1,
    
    -- Generation metadata
    generation_model VARCHAR(100),
    generation_prompt TEXT,
    generation_seed VARCHAR(100),
    generation_params JSONB,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT unique_hero_per_pose UNIQUE (pose_type, pose_direction) WHERE is_hero = true AND status = 'published'
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_kelly_assets_pose ON kelly_assets(pose_type, pose_direction);
CREATE INDEX IF NOT EXISTS idx_kelly_assets_status ON kelly_assets(status);
CREATE INDEX IF NOT EXISTS idx_kelly_assets_hero ON kelly_assets(is_hero) WHERE is_hero = true;

-- View for easy access to production assets
CREATE OR REPLACE VIEW kelly_production_assets AS
SELECT 
    id,
    pose_type,
    pose_direction,
    emotion,
    filename,
    CONCAT('https://kelly-assets.curiouskelly.com/', r2_key) as cdn_url,
    r2_key,
    created_at,
    published_at
FROM kelly_assets
WHERE status = 'published' AND is_hero = true;

-- Comment the table
COMMENT ON TABLE kelly_assets IS 'Manages all Kelly avatar images for the Curious Kelly platform';
COMMENT ON COLUMN kelly_assets.pose_type IS 'Core pose: idle, thinking, pointing_left, pointing_right, pointing_up, pointing_down, encouraging, hint, celebrating, supportive, proud, excited';
COMMENT ON COLUMN kelly_assets.pose_direction IS 'Optional direction modifier for poses that have variants';
COMMENT ON COLUMN kelly_assets.is_hero IS 'True if this is the primary/canonical version of this pose';
COMMENT ON COLUMN kelly_assets.status IS 'Workflow status: draft, review, approved, published, archived';



