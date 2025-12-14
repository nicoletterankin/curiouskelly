-- ============================================
-- KELLY MOTION LIBRARY (420 base clips)
-- ============================================
-- Stores phase-specific HeyGen videos per avatar_key (persona + age_bucket).
--
-- Expected usage:
-- - scripts/generate-motion-library.ts (service role key) upserts rows and updates video_url/status.
--
-- Safe to run multiple times.

-- gen_random_uuid() lives in pgcrypto on Supabase.
create extension if not exists pgcrypto;

create table if not exists public.kelly_motion_library (
  id uuid primary key default gen_random_uuid(),
  avatar_key text not null,
  persona text not null,
  age_bucket text not null,
  phase text not null,
  heygen_avatar_id text,
  video_id text,
  video_url text,
  status text default 'pending',
  duration double precision,
  error_message text,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  completed_at timestamptz,
  unique (avatar_key, phase)
);

create index if not exists idx_motion_avatar_key on public.kelly_motion_library(avatar_key);
create index if not exists idx_motion_phase on public.kelly_motion_library(phase);
create index if not exists idx_motion_status on public.kelly_motion_library(status);
create index if not exists idx_motion_age_bucket on public.kelly_motion_library(age_bucket);
create index if not exists idx_motion_completed on public.kelly_motion_library(completed_at desc);

-- Enable RLS + allow public reads for dashboards (anon key)
alter table public.kelly_motion_library enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'kelly_motion_library'
      and policyname = 'Allow public read access'
  ) then
    create policy "Allow public read access" on public.kelly_motion_library
      for select using (true);
  end if;
end $$;

