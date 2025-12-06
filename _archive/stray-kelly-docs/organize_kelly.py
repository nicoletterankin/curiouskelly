import os
import shutil
from pathlib import Path

BASE_DIR = "assets/kelly_canonical"
print("Creating canonical directory structure...")
PLAN = [
  {
    "original_path": "lessons/images/kelly-directors-chair-celebrating.png",
    "original_name": "kelly-directors-chair-celebrating.png",
    "new_category": "core/chair",
    "new_name": "kelly-chair-celebrating.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly-directors-chair-celebrating.png"
  },
  {
    "original_path": "lessons/images/kelly-directors-chair-curious.png",
    "original_name": "kelly-directors-chair-curious.png",
    "new_category": "core/chair",
    "new_name": "kelly-chair-curious.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly-directors-chair-curious.png"
  },
  {
    "original_path": "lessons/images/kelly-directors-chair-explaining.png",
    "original_name": "kelly-directors-chair-explaining.png",
    "new_category": "core/chair",
    "new_name": "kelly-chair-explaining.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly-directors-chair-explaining.png"
  },
  {
    "original_path": "lessons/images/kelly-directors-chair-listening.png",
    "original_name": "kelly-directors-chair-listening.png",
    "new_category": "core/chair",
    "new_name": "kelly-chair-listening.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly-directors-chair-listening.png"
  },
  {
    "original_path": "lessons/images/kelly-directors-chair-wisdom.png",
    "original_name": "kelly-directors-chair-wisdom.png",
    "new_category": "core/chair",
    "new_name": "kelly-chair-wisdom.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly-directors-chair-wisdom.png"
  },
  {
    "original_path": "lessons/Curious Kelly in final pose in Chair - UI elements will go on the side rails.png",
    "original_name": "Curious Kelly in final pose in Chair - UI elements will go on the side rails.png",
    "new_category": "junk_drawer",
    "new_name": "REVIEW-kelly-chair-ui-mockup.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/Curious Kelly in final pose in Chair - UI elements will go on the side rails.png"
  },
  {
    "original_path": "lessons/curious kelly.PNG",
    "original_name": "curious kelly.PNG",
    "new_category": "junk_drawer",
    "new_name": "REVIEW-kelly-portrait-legacy.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/curious kelly.PNG"
  },
  {
    "original_path": "projects/Kelly/assets/renders/kelly_expression_front_studio_neutral_v001.png",
    "original_name": "kelly_expression_front_studio_neutral_v001.png",
    "new_category": "junk_drawer",
    "new_name": "kelly-expression-front-studio-neutral.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_expression_front_studio_neutral_v001.png"
  },
  {
    "original_path": "projects/Kelly/assets/renders/kelly_hair_plate_front_backlit_edge_v001.png",
    "original_name": "kelly_hair_plate_front_backlit_edge_v001.png",
    "new_category": "junk_drawer",
    "new_name": "kelly-hair-plate-front-backlit-edge.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_hair_plate_front_backlit_edge_v001.png"
  },
  {
    "original_path": "synthetic_tts/kelly_directors_chair_8k_light.png",
    "original_name": "kelly_directors_chair_8k_light.png",
    "new_category": "junk_drawer",
    "new_name": "kelly-directors-chair-8k-light.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_directors_chair_8k_light.png"
  },
  {
    "original_path": "synthetic_tts/kelly_front_square_8k_transparent.png",
    "original_name": "kelly_front_square_8k_transparent.png",
    "new_category": "junk_drawer",
    "new_name": "kelly-front-square-8k-transparent.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_front_square_8k_transparent.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_close_up_portrait_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_15_close_up_portrait_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-closeup-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_close_up_portrait_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_close_up_portrait_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_15_close_up_portrait_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-closeup-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_close_up_portrait_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_close_up_portrait_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_15_close_up_portrait_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-closeup-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_close_up_portrait_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_front_facing_lean_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_15_front_facing_lean_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-front-lean-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_front_facing_lean_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_front_facing_lean_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_15_front_facing_lean_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-front-lean-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_front_facing_lean_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_front_facing_lean_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_15_front_facing_lean_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-front-lean-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_front_facing_lean_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_full_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_15_full_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-fullbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_full_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_full_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_15_full_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-fullbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_full_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_full_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_15_full_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-fullbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_full_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_upper_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_15_upper_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-upperbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_upper_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_upper_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_15_upper_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-upperbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_upper_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_15_upper_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_15_upper_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age15-upperbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_15_upper_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_close_up_portrait_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_27_close_up_portrait_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-closeup-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_close_up_portrait_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_close_up_portrait_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_27_close_up_portrait_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-closeup-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_close_up_portrait_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_close_up_portrait_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_27_close_up_portrait_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-closeup-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_close_up_portrait_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_front_facing_lean_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_27_front_facing_lean_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-front-lean-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_front_facing_lean_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_front_facing_lean_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_27_front_facing_lean_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-front-lean-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_front_facing_lean_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_front_facing_lean_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_27_front_facing_lean_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-front-lean-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_front_facing_lean_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_full_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_27_full_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-fullbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_full_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_full_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_27_full_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-fullbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_full_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_full_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_27_full_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-fullbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_full_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_upper_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_27_upper_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-upperbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_upper_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_upper_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_27_upper_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-upperbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_upper_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_27_upper_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_27_upper_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age27-upperbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_27_upper_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_close_up_portrait_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_3_close_up_portrait_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-closeup-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_close_up_portrait_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_close_up_portrait_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_3_close_up_portrait_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-closeup-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_close_up_portrait_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_close_up_portrait_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_3_close_up_portrait_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-closeup-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_close_up_portrait_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_front_facing_lean_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_3_front_facing_lean_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-front-lean-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_front_facing_lean_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_front_facing_lean_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_3_front_facing_lean_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-front-lean-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_front_facing_lean_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_front_facing_lean_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_3_front_facing_lean_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-front-lean-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_front_facing_lean_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_full_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_3_full_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-fullbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_full_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_full_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_3_full_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-fullbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_full_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_full_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_3_full_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-fullbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_full_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_upper_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_3_upper_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-upperbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_upper_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_upper_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_3_upper_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-upperbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_upper_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_3_upper_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_3_upper_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age3-upperbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_3_upper_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_close_up_portrait_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_48_close_up_portrait_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-closeup-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_close_up_portrait_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_close_up_portrait_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_48_close_up_portrait_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-closeup-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_close_up_portrait_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_close_up_portrait_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_48_close_up_portrait_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-closeup-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_close_up_portrait_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_front_facing_lean_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_48_front_facing_lean_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-front-lean-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_front_facing_lean_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_front_facing_lean_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_48_front_facing_lean_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-front-lean-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_front_facing_lean_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_front_facing_lean_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_48_front_facing_lean_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-front-lean-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_front_facing_lean_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_full_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_48_full_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-fullbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_full_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_full_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_48_full_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-fullbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_full_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_full_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_48_full_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-fullbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_full_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_upper_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_48_upper_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-upperbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_upper_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_upper_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_48_upper_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-upperbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_upper_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_48_upper_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_48_upper_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age48-upperbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_48_upper_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_close_up_portrait_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_82_close_up_portrait_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-closeup-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_close_up_portrait_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_close_up_portrait_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_82_close_up_portrait_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-closeup-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_close_up_portrait_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_close_up_portrait_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_82_close_up_portrait_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-closeup-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_close_up_portrait_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_front_facing_lean_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_82_front_facing_lean_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-front-lean-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_front_facing_lean_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_front_facing_lean_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_82_front_facing_lean_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-front-lean-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_front_facing_lean_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_front_facing_lean_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_82_front_facing_lean_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-front-lean-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_front_facing_lean_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_full_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_82_full_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-fullbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_full_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_full_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_82_full_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-fullbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_full_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_full_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_82_full_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-fullbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_full_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_upper_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_82_upper_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-upperbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_upper_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_upper_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_82_upper_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-upperbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_upper_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_82_upper_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_82_upper_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age82-upperbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_82_upper_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_close_up_portrait_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_9_close_up_portrait_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-closeup-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_close_up_portrait_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_close_up_portrait_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_9_close_up_portrait_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-closeup-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_close_up_portrait_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_close_up_portrait_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_9_close_up_portrait_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-closeup-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_close_up_portrait_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_front_facing_lean_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_9_front_facing_lean_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-front-lean-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_front_facing_lean_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_front_facing_lean_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_9_front_facing_lean_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-front-lean-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_front_facing_lean_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_front_facing_lean_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_9_front_facing_lean_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-front-lean-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_front_facing_lean_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_full_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_9_full_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-fullbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_full_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_full_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_9_full_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-fullbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_full_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_full_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_9_full_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-fullbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_full_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_upper_body_seated_front_studio_neutral_16x9.png",
    "original_name": "kelly_age_9_upper_body_seated_front_studio_neutral_16x9.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-upperbody-16x9.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_upper_body_seated_front_studio_neutral_16x9.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_upper_body_seated_front_studio_neutral_1x1.png",
    "original_name": "kelly_age_9_upper_body_seated_front_studio_neutral_1x1.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-upperbody-1x1.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_upper_body_seated_front_studio_neutral_1x1.png"
  },
  {
    "original_path": "projects/Kelly/assets/age_progressive/renders/kelly_age_9_upper_body_seated_front_studio_neutral_3x4.png",
    "original_name": "kelly_age_9_upper_body_seated_front_studio_neutral_3x4.png",
    "new_category": "marketing/age_variants",
    "new_name": "kelly-age9-upperbody-3x4.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_age_9_upper_body_seated_front_studio_neutral_3x4.png"
  },
  {
    "original_path": "public/images/kelly/kelly-upperbody-panelopen-christmas.png",
    "original_name": "kelly-upperbody-panelopen-christmas.png",
    "new_category": "marketing/seasonal",
    "new_name": "kelly-upperbody-panelopen-christmas.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly-upperbody-panelopen-christmas.png"
  },
  {
    "original_path": "iLearnStudio/projects/Kelly/Ref/kelly_front.png",
    "original_name": "kelly_front.png",
    "new_category": "reference/identity",
    "new_name": "kelly-ref-front-standard.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_front.png"
  },
  {
    "original_path": "iLearnStudio/projects/Kelly/Ref/kelly_profile.png",
    "original_name": "kelly_profile.png",
    "new_category": "reference/identity",
    "new_name": "kelly-profile.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_profile.png"
  },
  {
    "original_path": "iLearnStudio/projects/Kelly/Ref/kelly_three_quarter.png",
    "original_name": "kelly_three_quarter.png",
    "new_category": "reference/identity",
    "new_name": "kelly-three-quarter.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_three_quarter.png"
  },
  {
    "original_path": "projects/Kelly/assets/identity_contact_sheet.png",
    "original_name": "identity_contact_sheet.png",
    "new_category": "reference/identity",
    "new_name": "kelly-ref-contact-sheet.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/identity_contact_sheet.png"
  },
  {
    "original_path": "projects/Kelly/assets/renders/kelly_identity_front_studio_neutral_v001.png",
    "original_name": "kelly_identity_front_studio_neutral_v001.png",
    "new_category": "reference/identity",
    "new_name": "kelly-identity-front-studio-neutral.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_identity_front_studio_neutral_v001.png"
  },
  {
    "original_path": "projects/Kelly/assets/renders/kelly_identity_profile_studio_neutral_v001.png",
    "original_name": "kelly_identity_profile_studio_neutral_v001.png",
    "new_category": "reference/identity",
    "new_name": "kelly-identity-profile-studio-neutral.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_identity_profile_studio_neutral_v001.png"
  },
  {
    "original_path": "projects/Kelly/assets/renders/kelly_identity_three_quarter_studio_neutral_v001.png",
    "original_name": "kelly_identity_three_quarter_studio_neutral_v001.png",
    "new_category": "reference/identity",
    "new_name": "kelly-identity-three-quarter-studio-neutral.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_identity_three_quarter_studio_neutral_v001.png"
  },
  {
    "original_path": "projects/Kelly/Ref/headshot2-kelly-base169 101225.png",
    "original_name": "headshot2-kelly-base169 101225.png",
    "new_category": "reference/identity",
    "new_name": "kelly-headshot2-base169-101225.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/headshot2-kelly-base169 101225.png"
  },
  {
    "original_path": "projects/Kelly/Ref/kelly_headshot_4k.png",
    "original_name": "kelly_headshot_4k.png",
    "new_category": "reference/identity",
    "new_name": "kelly-headshot-4k.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_headshot_4k.png"
  },
  {
    "original_path": "projects/Kelly/Ref/kelly_headshot_extracted.png",
    "original_name": "kelly_headshot_extracted.png",
    "new_category": "reference/identity",
    "new_name": "kelly-headshot-extracted.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/kelly_headshot_extracted.png"
  },
  {
    "original_path": "projects/Kelly/CC5/HairPhysics/Fine_Strand_Noise.png",
    "original_name": "Fine_Strand_Noise.png",
    "new_category": "reference/texture",
    "new_name": "kelly-fine-strand-noise.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/Fine_Strand_Noise.png"
  },
  {
    "original_path": "projects/Kelly/CC5/HairPhysics/Kelly_Hair_PhysicsMap.png",
    "original_name": "Kelly_Hair_PhysicsMap.png",
    "new_category": "reference/texture",
    "new_name": "kelly-hair-physicsmap.png",
    "public_url": "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/images/kelly/Kelly_Hair_PhysicsMap.png"
  }
]


for item in PLAN:
    dest_dir = os.path.join(BASE_DIR, item['new_category'])
    os.makedirs(dest_dir, exist_ok=True)
    
    src = item['original_path']
    dst = os.path.join(dest_dir, item['new_name'])
    
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {item['new_name']}")
        else:
            print(f"MISSING: {src}")
    except Exception as e:
        print(f"Error copying {src}: {e}")
