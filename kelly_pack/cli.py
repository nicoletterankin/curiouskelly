"""
CLI interface for Kelly Asset Pack Generator
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from . import io_utils, crop_scale, matting, alpha_tools, composite, diffuse, physics_sheet, video_frame


def build_all(args):
    """
    Build all Kelly assets.
    """
    print("=" * 60)
    print("Kelly Asset Pack Generator - Build All")
    print("=" * 60)
    
    # Find input files
    chair_path = io_utils.find_first_existing(
        args.chair,
        args.chair_fallback,
        "kelly2-directors-chair.jpeg",
        "Kelly Source.jpeg"
    )
    
    portrait_path = io_utils.find_first_existing(
        args.portrait,
        "reference_kelly_image.png"
    )
    
    tight_portrait_path = io_utils.find_first_existing(
        args.tight_portrait,
        "ChatGPT Image Oct 9, 2025, 01_17_55 PM.png"
    )
    
    video_path = io_utils.find_first_existing(
        args.video,
        "Avatar IV Video (1).mp4"
    )
    
    # Validation
    if not chair_path:
        print("ERROR: No chair image found. Please provide --chair or place kelly2-directors-chair.jpeg in current directory.")
        return 1
    
    print(f"\nInput files:")
    print(f"  Chair: {chair_path}")
    if portrait_path:
        print(f"  Portrait: {portrait_path}")
    if tight_portrait_path:
        print(f"  Tight portrait: {tight_portrait_path}")
    if video_path:
        print(f"  Video: {video_path}")
    
    # Setup
    io_utils.ensure_dir(args.outdir)
    device = "cuda" if args.device == "cuda" else "cpu"
    use_model = not args.no_torch
    
    # Step 1: Load and prepare chair image (16:9 hero)
    print("\n" + "=" * 60)
    print("Step 1: Processing 16:9 Hero (Chair)")
    print("=" * 60)
    
    chair_img = io_utils.load_image(chair_path, mode="RGB")
    if chair_img is None:
        print(f"ERROR: Could not load {chair_path}")
        return 1
    
    # Prepare 16:9 at 8K
    hero_rgb = crop_scale.prepare_16_9_hero(chair_img, 7680, 4320)
    
    # Generate alpha (process at compute-friendly size, then upsample)
    print("\nGenerating alpha matte...")
    # Work at 2K for matting
    hero_rgb_2k = crop_scale.resize_lanczos(hero_rgb, (2560, 1440), mode="RGB")
    base_alpha_2k = matting.generate_alpha(hero_rgb_2k, use_model=use_model, device=device, weights_dir=args.weights_dir)
    
    # Upsample to 8K with guided filter
    print("Upsampling alpha to 8K with edge preservation...")
    base_alpha = matting.guided_upsample_alpha(base_alpha_2k, hero_rgb, (7680, 4320))
    
    # Generate alpha variants
    print("Generating soft/tight/edge alpha variants...")
    alpha_soft, alpha_tight, alpha_edge = alpha_tools.generate_alpha_variants(
        base_alpha,
        soft_blur=args.soft_blur,
        soft_bias=args.soft_bias,
        tight_blur=args.tight_blur,
        tight_bias=args.tight_bias,
        tight_erode=args.tight_erode
    )
    
    # Save alpha utilities
    io_utils.save_image((alpha_soft * 255).astype('uint8'), 
                       f"{args.outdir}/kelly_alpha_soft_8k.png", mode="L")
    io_utils.save_image((alpha_tight * 255).astype('uint8'), 
                       f"{args.outdir}/kelly_alpha_tight_8k.png", mode="L")
    io_utils.save_image((alpha_edge * 255).astype('uint8'), 
                       f"{args.outdir}/kelly_hair_edge_matte_8k.png", mode="L")
    
    # Step 2: Create transparent hero
    print("\n" + "=" * 60)
    print("Step 2: Creating Transparent Hero")
    print("=" * 60)
    
    hero_rgba = io_utils.np.dstack([hero_rgb, (alpha_tight * 255).astype('uint8')])
    io_utils.save_image(hero_rgba, f"{args.outdir}/kelly_directors_chair_8k_transparent.png", mode="RGBA")
    
    # Step 3: Create dark hero
    print("\n" + "=" * 60)
    print("Step 3: Creating Dark Hero")
    print("=" * 60)
    
    dark_hero = composite.create_dark_hero(hero_rgb, alpha_tight, args.grad_top, args.grad_bottom)
    io_utils.save_image(dark_hero, f"{args.outdir}/kelly_directors_chair_8k_dark.png", mode="RGB")
    
    # Step 4: Create diffuse neutrals (16:9)
    print("\n" + "=" * 60)
    print("Step 4: Creating Diffuse Neutral (16:9)")
    print("=" * 60)
    
    chair_diffuse = diffuse.neutralize_diffuse(hero_rgb, args.contrast_flatten)
    io_utils.save_image(chair_diffuse, f"{args.outdir}/kelly_chair_diffuse_neutral_8k.png", mode="RGB")
    
    # Step 5: Square sprite
    print("\n" + "=" * 60)
    print("Step 5: Creating Square Sprite")
    print("=" * 60)
    
    # Choose best portrait source
    sprite_source = io_utils.find_first_existing(tight_portrait_path, portrait_path, chair_path)
    if sprite_source:
        sprite_img = io_utils.load_image(sprite_source, mode="RGB")
        
        # Generate alpha for sprite
        print("Generating sprite alpha...")
        sprite_alpha = matting.generate_alpha(sprite_img, use_model=use_model, device=device, weights_dir=args.weights_dir)
        sprite_alpha_soft = alpha_tools.generate_soft_alpha(sprite_alpha, args.soft_blur, args.soft_bias)
        
        # Create square canvas
        sprite_rgb, sprite_alpha_canvas = crop_scale.prepare_square_sprite(
            sprite_img, sprite_alpha_soft, 8192, args.padding_frac
        )
        
        sprite_rgba = io_utils.np.dstack([sprite_rgb, (sprite_alpha_canvas * 255).astype('uint8')])
        io_utils.save_image(sprite_rgba, f"{args.outdir}/kelly_front_square_8k_transparent.png", mode="RGBA")
        
        # Diffuse neutral (square)
        print("Creating square diffuse neutral...")
        sprite_diffuse = diffuse.neutralize_diffuse(sprite_rgb, args.contrast_flatten)
        io_utils.save_image(sprite_diffuse, f"{args.outdir}/kelly_diffuse_neutral_8k.png", mode="RGB")
    else:
        print("Warning: No portrait source found for square sprite")
    
    # Step 6: Physics reference
    print("\n" + "=" * 60)
    print("Step 6: Creating Physics Reference Sheet")
    print("=" * 60)
    
    physics_sheet.generate_physics_pdf(f"{args.outdir}/kelly_physics_reference_sheet.pdf")
    
    # Step 7: Optional video frame
    if video_path:
        print("\n" + "=" * 60)
        print("Step 7: Extracting Video Mid-Frame")
        print("=" * 60)
        
        video_frame_img = video_frame.extract_midframe(video_path, 2.0)
        if video_frame_img is not None:
            # Crop and scale to 8K 16:9
            video_frame_8k = crop_scale.prepare_16_9_hero(video_frame_img, 7680, 4320)
            io_utils.save_image(video_frame_8k, f"{args.outdir}/kelly_video_midframe_8k.png", mode="RGB")
    
    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs saved to: {args.outdir}")
    print("\nGenerated files:")
    print("  1. kelly_directors_chair_8k_transparent.png (7680x4320 RGBA)")
    print("  2. kelly_directors_chair_8k_dark.png (7680x4320 RGB)")
    print("  3. kelly_front_square_8k_transparent.png (8192x8192 RGBA)")
    print("  4. kelly_diffuse_neutral_8k.png (8192x8192 RGB)")
    print("  5. kelly_chair_diffuse_neutral_8k.png (7680x4320 RGB)")
    print("  6. kelly_alpha_soft_8k.png (7680x4320 L)")
    print("  7. kelly_alpha_tight_8k.png (7680x4320 L)")
    print("  8. kelly_hair_edge_matte_8k.png (7680x4320 L)")
    print("  9. kelly_physics_reference_sheet.pdf")
    if video_path:
        print(" 10. kelly_video_midframe_8k.png (7680x4320 RGB)")
    
    return 0


def hair_only(args):
    """Regenerate hair alphas only."""
    print("Regenerating hair alphas...")
    
    chair_path = io_utils.find_first_existing(args.chair, "kelly2-directors-chair.jpeg")
    if not chair_path:
        print("ERROR: No chair image found")
        return 1
    
    chair_img = io_utils.load_image(chair_path, mode="RGB")
    hero_rgb = crop_scale.prepare_16_9_hero(chair_img, 7680, 4320)
    
    device = "cuda" if args.device == "cuda" else "cpu"
    use_model = not args.no_torch
    
    hero_rgb_2k = crop_scale.resize_lanczos(hero_rgb, (2560, 1440), mode="RGB")
    base_alpha_2k = matting.generate_alpha(hero_rgb_2k, use_model=use_model, device=device, weights_dir=args.weights_dir)
    base_alpha = matting.guided_upsample_alpha(base_alpha_2k, hero_rgb, (7680, 4320))
    
    alpha_soft, alpha_tight, alpha_edge = alpha_tools.generate_alpha_variants(
        base_alpha,
        soft_blur=args.soft_blur,
        soft_bias=args.soft_bias,
        tight_blur=args.tight_blur,
        tight_bias=args.tight_bias,
        tight_erode=args.tight_erode
    )
    
    io_utils.save_image((alpha_soft * 255).astype('uint8'), 
                       f"{args.outdir}/kelly_alpha_soft_8k.png", mode="L")
    io_utils.save_image((alpha_tight * 255).astype('uint8'), 
                       f"{args.outdir}/kelly_alpha_tight_8k.png", mode="L")
    io_utils.save_image((alpha_edge * 255).astype('uint8'), 
                       f"{args.outdir}/kelly_hair_edge_matte_8k.png", mode="L")
    
    print("Done!")
    return 0


def dark_hero_only(args):
    """Regenerate dark hero only."""
    print("Regenerating dark hero...")
    
    # Load existing transparent
    trans_path = f"{args.outdir}/kelly_directors_chair_8k_transparent.png"
    if not io_utils.os.path.exists(trans_path):
        print(f"ERROR: {trans_path} not found. Run 'build' first.")
        return 1
    
    rgba = io_utils.load_image(trans_path, mode="RGBA")
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3].astype('float32') / 255.0
    
    dark_hero = composite.create_dark_hero(rgb, alpha, args.grad_top, args.grad_bottom)
    io_utils.save_image(dark_hero, f"{args.outdir}/kelly_directors_chair_8k_dark.png", mode="RGB")
    
    print("Done!")
    return 0


def sprite_only(args):
    """Regenerate square sprite only."""
    print("Regenerating square sprite...")
    
    portrait_path = io_utils.find_first_existing(args.portrait, "reference_kelly_image.png")
    if not portrait_path:
        print("ERROR: No portrait image found")
        return 1
    
    sprite_img = io_utils.load_image(portrait_path, mode="RGB")
    
    device = "cuda" if args.device == "cuda" else "cpu"
    use_model = not args.no_torch
    
    sprite_alpha = matting.generate_alpha(sprite_img, use_model=use_model, device=device, weights_dir=args.weights_dir)
    sprite_alpha_soft = alpha_tools.generate_soft_alpha(sprite_alpha, args.soft_blur, args.soft_bias)
    
    sprite_rgb, sprite_alpha_canvas = crop_scale.prepare_square_sprite(
        sprite_img, sprite_alpha_soft, 8192, args.padding_frac
    )
    
    sprite_rgba = io_utils.np.dstack([sprite_rgb, (sprite_alpha_canvas * 255).astype('uint8')])
    io_utils.save_image(sprite_rgba, f"{args.outdir}/kelly_front_square_8k_transparent.png", mode="RGBA")
    
    sprite_diffuse = diffuse.neutralize_diffuse(sprite_rgb, args.contrast_flatten)
    io_utils.save_image(sprite_diffuse, f"{args.outdir}/kelly_diffuse_neutral_8k.png", mode="RGB")
    
    print("Done!")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Kelly Asset Pack Generator - 8K Digital Human Assets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build all assets")
    build_parser.add_argument("--chair", help="Chair image path")
    build_parser.add_argument("--chair-fallback", help="Fallback chair image")
    build_parser.add_argument("--portrait", help="Portrait image for sprite")
    build_parser.add_argument("--tight-portrait", help="Tight portrait for sprite")
    build_parser.add_argument("--video", help="Optional video file")
    build_parser.add_argument("--outdir", default=".", help="Output directory")
    build_parser.add_argument("--soft-blur", type=float, default=2.0, help="Soft alpha blur radius")
    build_parser.add_argument("--soft-bias", type=float, default=0.05, help="Soft alpha bias")
    build_parser.add_argument("--tight-blur", type=float, default=1.0, help="Tight alpha blur radius")
    build_parser.add_argument("--tight-bias", type=float, default=-0.03, help="Tight alpha bias")
    build_parser.add_argument("--tight-erode", type=int, default=1, help="Tight alpha erosion size")
    build_parser.add_argument("--grad-top", default="#22262A", help="Dark gradient top color")
    build_parser.add_argument("--grad-bottom", default="#080808", help="Dark gradient bottom color")
    build_parser.add_argument("--padding-frac", type=float, default=0.10, help="Square sprite padding")
    build_parser.add_argument("--contrast-flatten", type=float, default=0.15, help="Diffuse contrast flatten")
    build_parser.add_argument("--no-torch", action="store_true", help="Force heuristic matting")
    build_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Compute device")
    build_parser.add_argument("--weights-dir", default="./weights", help="Model weights directory")
    build_parser.add_argument("--keep-intermediates", action="store_true", help="Keep debug images")
    
    # Hair command
    hair_parser = subparsers.add_parser("hair", help="Regenerate hair alphas only")
    hair_parser.add_argument("--chair", help="Chair image path")
    hair_parser.add_argument("--outdir", default=".", help="Output directory")
    hair_parser.add_argument("--soft-blur", type=float, default=2.0)
    hair_parser.add_argument("--soft-bias", type=float, default=0.05)
    hair_parser.add_argument("--tight-blur", type=float, default=1.0)
    hair_parser.add_argument("--tight-bias", type=float, default=-0.03)
    hair_parser.add_argument("--tight-erode", type=int, default=1)
    hair_parser.add_argument("--no-torch", action="store_true")
    hair_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    hair_parser.add_argument("--weights-dir", default="./weights")
    
    # Dark hero command
    dark_parser = subparsers.add_parser("dark-hero", help="Regenerate dark hero only")
    dark_parser.add_argument("--outdir", default=".", help="Output directory")
    dark_parser.add_argument("--grad-top", default="#22262A")
    dark_parser.add_argument("--grad-bottom", default="#080808")
    
    # Sprite command
    sprite_parser = subparsers.add_parser("sprite", help="Regenerate square sprite only")
    sprite_parser.add_argument("--portrait", help="Portrait image")
    sprite_parser.add_argument("--outdir", default=".", help="Output directory")
    sprite_parser.add_argument("--soft-blur", type=float, default=2.0)
    sprite_parser.add_argument("--soft-bias", type=float, default=0.05)
    sprite_parser.add_argument("--padding-frac", type=float, default=0.10)
    sprite_parser.add_argument("--contrast-flatten", type=float, default=0.15)
    sprite_parser.add_argument("--no-torch", action="store_true")
    sprite_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    sprite_parser.add_argument("--weights-dir", default="./weights")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to command
    if args.command == "build":
        return build_all(args)
    elif args.command == "hair":
        return hair_only(args)
    elif args.command == "dark-hero":
        return dark_hero_only(args)
    elif args.command == "sprite":
        return sprite_only(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


