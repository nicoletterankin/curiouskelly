-- ═══════════════════════════════════════════════════════════════════════════
-- MIGRATION 003b: POPULATE THUMBNAIL SLUGS
-- ═══════════════════════════════════════════════════════════════════════════
-- 
-- This migration populates the thumbnail_slug column in core_lessons
-- based on the actual files that exist in /kelly/thumbnails/raw/
--
-- Generated from file scan on December 3, 2025
-- ═══════════════════════════════════════════════════════════════════════════

-- Clear any existing values first (optional, remove if you want to preserve)
-- UPDATE core_lessons SET thumbnail_slug = NULL;

-- ═══════════════════════════════════════════════════════════════════════════
-- THUMBNAIL SLUG MAPPINGS
-- Extracted from: public/kelly/thumbnails/raw/*.png
-- ═══════════════════════════════════════════════════════════════════════════

UPDATE core_lessons SET thumbnail_slug = 'starting-fresh' WHERE day_number = 1;
UPDATE core_lessons SET thumbnail_slug = 'the-three-lives-of-water' WHERE day_number = 2;
UPDATE core_lessons SET thumbnail_slug = 'where-clouds-come-from' WHERE day_number = 3;
UPDATE core_lessons SET thumbnail_slug = 'how-light-travels' WHERE day_number = 4;
UPDATE core_lessons SET thumbnail_slug = 'how-sound-moves' WHERE day_number = 5;
UPDATE core_lessons SET thumbnail_slug = 'whats-inside-a-seed' WHERE day_number = 6;
UPDATE core_lessons SET thumbnail_slug = 'what-stars-are-made-of' WHERE day_number = 7;
UPDATE core_lessons SET thumbnail_slug = 'what-makes-a-real-friend' WHERE day_number = 8;
UPDATE core_lessons SET thumbnail_slug = 'how-kindness-spreads' WHERE day_number = 9;
UPDATE core_lessons SET thumbnail_slug = 'the-art-of-really-listening' WHERE day_number = 10;
UPDATE core_lessons SET thumbnail_slug = 'why-patience-pays-off' WHERE day_number = 11;
UPDATE core_lessons SET thumbnail_slug = 'how-gratitude-changes-you' WHERE day_number = 12;
UPDATE core_lessons SET thumbnail_slug = 'what-courage-really-means' WHERE day_number = 13;
UPDATE core_lessons SET thumbnail_slug = 'the-power-of-questions' WHERE day_number = 14;
UPDATE core_lessons SET thumbnail_slug = 'how-your-body-stays-balanced' WHERE day_number = 15;
UPDATE core_lessons SET thumbnail_slug = 'what-makes-music-feel' WHERE day_number = 16;
UPDATE core_lessons SET thumbnail_slug = 'why-mistakes-matter' WHERE day_number = 17;
UPDATE core_lessons SET thumbnail_slug = 'how-plants-eat-sunlight' WHERE day_number = 18;
UPDATE core_lessons SET thumbnail_slug = 'how-energy-changes-form' WHERE day_number = 19;
UPDATE core_lessons SET thumbnail_slug = 'why-we-need-sleep' WHERE day_number = 20;
UPDATE core_lessons SET thumbnail_slug = 'how-to-disagree-well' WHERE day_number = 21;
UPDATE core_lessons SET thumbnail_slug = 'what-gravity-really-does' WHERE day_number = 22;
UPDATE core_lessons SET thumbnail_slug = 'patterns-are-everywhere' WHERE day_number = 23;
UPDATE core_lessons SET thumbnail_slug = 'how-memory-works' WHERE day_number = 24;
UPDATE core_lessons SET thumbnail_slug = 'why-music-moves-us' WHERE day_number = 25;
UPDATE core_lessons SET thumbnail_slug = 'the-power-of-good-questions' WHERE day_number = 26;
UPDATE core_lessons SET thumbnail_slug = 'how-ice-shapes-land' WHERE day_number = 27;
UPDATE core_lessons SET thumbnail_slug = 'how-memories-are-made' WHERE day_number = 28;
UPDATE core_lessons SET thumbnail_slug = 'the-hidden-life-of-soil' WHERE day_number = 29;
UPDATE core_lessons SET thumbnail_slug = 'why-everything-changes' WHERE day_number = 30;
UPDATE core_lessons SET thumbnail_slug = 'how-change-happens' WHERE day_number = 31;
UPDATE core_lessons SET thumbnail_slug = 'the-moon-and-the-tides' WHERE day_number = 32;
UPDATE core_lessons SET thumbnail_slug = 'what-gravity-actually-does' WHERE day_number = 33;
UPDATE core_lessons SET thumbnail_slug = 'how-magnets-work' WHERE day_number = 34;
UPDATE core_lessons SET thumbnail_slug = 'how-electricity-flows' WHERE day_number = 35;
UPDATE core_lessons SET thumbnail_slug = 'what-fire-really-is' WHERE day_number = 36;
UPDATE core_lessons SET thumbnail_slug = 'why-ice-floats' WHERE day_number = 37;
UPDATE core_lessons SET thumbnail_slug = 'what-makes-wind-blow' WHERE day_number = 38;
UPDATE core_lessons SET thumbnail_slug = 'where-rain-comes-from' WHERE day_number = 39;
UPDATE core_lessons SET thumbnail_slug = 'what-causes-thunder' WHERE day_number = 40;
UPDATE core_lessons SET thumbnail_slug = 'how-rainbows-form' WHERE day_number = 41;
UPDATE core_lessons SET thumbnail_slug = 'why-seasons-change' WHERE day_number = 42;
UPDATE core_lessons SET thumbnail_slug = 'why-we-have-day-and-night' WHERE day_number = 43;
UPDATE core_lessons SET thumbnail_slug = 'how-shadows-work' WHERE day_number = 44;
UPDATE core_lessons SET thumbnail_slug = 'why-mirrors-reflect' WHERE day_number = 45;
UPDATE core_lessons SET thumbnail_slug = 'how-sound-bounces-back' WHERE day_number = 46;
UPDATE core_lessons SET thumbnail_slug = 'how-waves-carry-energy' WHERE day_number = 47;
UPDATE core_lessons SET thumbnail_slug = 'the-science-of-bubbles' WHERE day_number = 48;
UPDATE core_lessons SET thumbnail_slug = 'how-crystals-form' WHERE day_number = 49;
UPDATE core_lessons SET thumbnail_slug = 'stories-trapped-in-stone' WHERE day_number = 50;
UPDATE core_lessons SET thumbnail_slug = 'when-dinosaurs-ruled' WHERE day_number = 51;
UPDATE core_lessons SET thumbnail_slug = 'whats-inside-a-volcano' WHERE day_number = 52;
UPDATE core_lessons SET thumbnail_slug = 'why-the-ground-shakes' WHERE day_number = 53;
UPDATE core_lessons SET thumbnail_slug = 'how-mountains-are-made' WHERE day_number = 54;
UPDATE core_lessons SET thumbnail_slug = 'the-deep-ocean-mystery' WHERE day_number = 55;
UPDATE core_lessons SET thumbnail_slug = 'how-rivers-shape-the-land' WHERE day_number = 56;
UPDATE core_lessons SET thumbnail_slug = 'where-lakes-come-from' WHERE day_number = 57;
UPDATE core_lessons SET thumbnail_slug = 'life-in-the-desert' WHERE day_number = 58;
UPDATE core_lessons SET thumbnail_slug = 'the-secret-life-of-forests' WHERE day_number = 59;
UPDATE core_lessons SET thumbnail_slug = 'why-jungles-are-so-alive' WHERE day_number = 60;
UPDATE core_lessons SET thumbnail_slug = 'the-power-of-grass' WHERE day_number = 61;
UPDATE core_lessons SET thumbnail_slug = 'why-wetlands-matter' WHERE day_number = 62;
UPDATE core_lessons SET thumbnail_slug = 'cities-under-the-sea' WHERE day_number = 63;
UPDATE core_lessons SET thumbnail_slug = 'worlds-without-light' WHERE day_number = 64;
UPDATE core_lessons SET thumbnail_slug = 'how-islands-are-born' WHERE day_number = 65;
UPDATE core_lessons SET thumbnail_slug = 'whats-living-in-the-dirt' WHERE day_number = 66;
UPDATE core_lessons SET thumbnail_slug = 'the-stories-rocks-tell' WHERE day_number = 67;
UPDATE core_lessons SET thumbnail_slug = 'earths-hidden-treasures' WHERE day_number = 68;
UPDATE core_lessons SET thumbnail_slug = 'how-gems-are-made' WHERE day_number = 69;
UPDATE core_lessons SET thumbnail_slug = 'where-metals-come-from' WHERE day_number = 70;
UPDATE core_lessons SET thumbnail_slug = 'whats-in-the-air-you-breathe' WHERE day_number = 71;
UPDATE core_lessons SET thumbnail_slug = 'why-we-need-oxygen' WHERE day_number = 72;
UPDATE core_lessons SET thumbnail_slug = 'carbon-is-everywhere' WHERE day_number = 73;
UPDATE core_lessons SET thumbnail_slug = 'the-gas-you-dont-notice' WHERE day_number = 74;
UPDATE core_lessons SET thumbnail_slug = 'the-simplest-element' WHERE day_number = 75;
UPDATE core_lessons SET thumbnail_slug = 'building-blocks-of-everything' WHERE day_number = 76;
UPDATE core_lessons SET thumbnail_slug = 'when-atoms-connect' WHERE day_number = 77;
UPDATE core_lessons SET thumbnail_slug = 'the-tiny-units-of-life' WHERE day_number = 78;
UPDATE core_lessons SET thumbnail_slug = 'your-bodys-instruction-manual' WHERE day_number = 79;
UPDATE core_lessons SET thumbnail_slug = 'what-blood-does-all-day' WHERE day_number = 80;
UPDATE core_lessons SET thumbnail_slug = 'how-taste-works' WHERE day_number = 81;

-- ═══════════════════════════════════════════════════════════════════════════
-- VERIFICATION
-- ═══════════════════════════════════════════════════════════════════════════

SELECT 
  'Thumbnail slugs populated' as status,
  COUNT(*) FILTER (WHERE thumbnail_slug IS NOT NULL) as with_slug,
  COUNT(*) FILTER (WHERE thumbnail_slug IS NULL) as without_slug,
  COUNT(*) as total
FROM core_lessons;

-- Show the first 15 for verification
SELECT day_number, topic, thumbnail_slug 
FROM core_lessons 
WHERE day_number <= 15 
ORDER BY day_number;



