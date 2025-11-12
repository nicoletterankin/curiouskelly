"""
Test asset generation: validate shapes, formats, and file existence
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kelly_pack import io_utils, crop_scale, alpha_tools, composite, diffuse


class TestShapesAndFormats:
    """Test output shapes and formats."""
    
    def test_16_9_dimensions(self):
        """Test 16:9 hero dimensions."""
        # Create test image
        test_img = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        result = crop_scale.prepare_16_9_hero(test_img, 7680, 4320)
        
        assert result.shape == (4320, 7680, 3), f"Expected (4320, 7680, 3), got {result.shape}"
    
    def test_square_dimensions(self):
        """Test square sprite dimensions."""
        # Create test image and alpha
        test_img = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
        test_alpha = np.random.rand(1000, 800).astype(np.float32)
        
        rgb_canvas, alpha_canvas = crop_scale.prepare_square_sprite(test_img, test_alpha, 8192, 0.10)
        
        assert rgb_canvas.shape == (8192, 8192, 3), f"Expected (8192, 8192, 3), got {rgb_canvas.shape}"
        assert alpha_canvas.shape == (8192, 8192), f"Expected (8192, 8192), got {alpha_canvas.shape}"
    
    def test_alpha_variants_differ(self):
        """Test that soft and tight alphas are different."""
        base_alpha = np.random.rand(1000, 1000).astype(np.float32)
        
        alpha_soft, alpha_tight, alpha_edge = alpha_tools.generate_alpha_variants(base_alpha)
        
        # Soft and tight should differ
        assert not np.allclose(alpha_soft, alpha_tight), "Soft and tight alphas should differ"
        
        # Edge should be non-zero
        assert np.any(alpha_edge > 0), "Edge matte should have non-zero values"
        
        # Edge should approximately equal soft - tight
        expected_edge = np.clip(alpha_soft - alpha_tight, 0, 1)
        assert np.allclose(alpha_edge, expected_edge, atol=0.01), "Edge matte should be soft - tight"
    
    def test_gradient_generation(self):
        """Test gradient background generation."""
        gradient = composite.create_vertical_gradient(7680, 4320, "#22262A", "#080808")
        
        assert gradient.shape == (4320, 7680, 3), f"Expected (4320, 7680, 3), got {gradient.shape}"
        
        # Top should be brighter than bottom
        top_brightness = np.mean(gradient[0, :, :])
        bottom_brightness = np.mean(gradient[-1, :, :])
        assert top_brightness > bottom_brightness, "Top should be brighter than bottom"
    
    def test_diffuse_neutralization(self):
        """Test diffuse neutralization."""
        test_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        result = diffuse.neutralize_diffuse(test_img, 0.15)
        
        assert result.shape == test_img.shape, "Shape should be preserved"
        assert result.dtype == np.uint8, "Should be uint8"


class TestFileGeneration:
    """Test full pipeline file generation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample test image."""
        # Create simple test image (white background, dark subject)
        img = np.ones((2000, 3000, 3), dtype=np.uint8) * 240  # Light background
        
        # Add a dark rectangle (subject)
        img[500:1500, 1000:2000] = [50, 60, 70]
        
        # Add some "hair-like" edges (gradual falloff)
        for i in range(100):
            alpha_val = i / 100.0
            color = np.array([50, 60, 70]) * alpha_val + 240 * (1 - alpha_val)
            img[400+i:500, 1000:2000] = color.astype(np.uint8)
        
        path = os.path.join(temp_dir, "test_chair.jpg")
        Image.fromarray(img).save(path)
        return path
    
    def test_build_outputs_exist(self, temp_dir, sample_image):
        """Test that build generates all expected files."""
        from kelly_pack.cli import build_all
        from argparse import Namespace
        
        # Create args
        args = Namespace(
            chair=sample_image,
            chair_fallback=None,
            portrait=sample_image,
            tight_portrait=None,
            video=None,
            outdir=temp_dir,
            soft_blur=2.0,
            soft_bias=0.05,
            tight_blur=1.0,
            tight_bias=-0.03,
            tight_erode=1,
            grad_top="#22262A",
            grad_bottom="#080808",
            padding_frac=0.10,
            contrast_flatten=0.15,
            no_torch=True,  # Force heuristic for testing
            device="cpu",
            weights_dir="./weights",
            keep_intermediates=False
        )
        
        # Run build (with smaller sizes for speed)
        # Temporarily patch the target sizes
        original_prepare = crop_scale.prepare_16_9_hero
        
        def mock_prepare(img, w=7680, h=4320):
            # Use smaller size for testing
            return original_prepare(img, 768, 432)
        
        crop_scale.prepare_16_9_hero = mock_prepare
        
        try:
            result = build_all(args)
            assert result == 0, "Build should succeed"
        finally:
            crop_scale.prepare_16_9_hero = original_prepare
        
        # Check that key files exist (not checking all 10 since we skipped video)
        expected_files = [
            "kelly_alpha_soft_8k.png",
            "kelly_alpha_tight_8k.png",
            "kelly_hair_edge_matte_8k.png",
            "kelly_directors_chair_8k_transparent.png",
            "kelly_directors_chair_8k_dark.png",
        ]
        
        for filename in expected_files:
            path = os.path.join(temp_dir, filename)
            assert os.path.exists(path), f"{filename} should exist"
    
    def test_output_image_sizes(self, temp_dir, sample_image):
        """Test that output images have correct dimensions."""
        from kelly_pack.cli import build_all
        from argparse import Namespace
        
        args = Namespace(
            chair=sample_image,
            chair_fallback=None,
            portrait=sample_image,
            tight_portrait=None,
            video=None,
            outdir=temp_dir,
            soft_blur=2.0,
            soft_bias=0.05,
            tight_blur=1.0,
            tight_bias=-0.03,
            tight_erode=1,
            grad_top="#22262A",
            grad_bottom="#080808",
            padding_frac=0.10,
            contrast_flatten=0.15,
            no_torch=True,
            device="cpu",
            weights_dir="./weights",
            keep_intermediates=False
        )
        
        # Use smaller test sizes
        original_prepare = crop_scale.prepare_16_9_hero
        original_sprite = crop_scale.prepare_square_sprite
        
        def mock_prepare(img, w=7680, h=4320):
            return original_prepare(img, 768, 432)
        
        def mock_sprite(img, alpha, size=8192, padding=0.10):
            return original_sprite(img, alpha, 512, padding)
        
        crop_scale.prepare_16_9_hero = mock_prepare
        crop_scale.prepare_square_sprite = mock_sprite
        
        try:
            build_all(args)
        finally:
            crop_scale.prepare_16_9_hero = original_prepare
            crop_scale.prepare_square_sprite = original_sprite
        
        # Check 16:9 files
        trans_img = Image.open(os.path.join(temp_dir, "kelly_directors_chair_8k_transparent.png"))
        assert trans_img.size == (768, 432), f"Expected (768, 432), got {trans_img.size}"
        assert trans_img.mode == "RGBA", f"Expected RGBA, got {trans_img.mode}"
        
        # Check square file
        sprite_img = Image.open(os.path.join(temp_dir, "kelly_front_square_8k_transparent.png"))
        assert sprite_img.size == (512, 512), f"Expected (512, 512), got {sprite_img.size}"
        assert sprite_img.mode == "RGBA", f"Expected RGBA, got {sprite_img.mode}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


