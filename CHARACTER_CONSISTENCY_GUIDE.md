# How to Lock In Kelly's Character Consistency

## Summary

I've created a character reference system that ensures all future asset generation maintains Kelly's exact appearance and photorealistic style. Here's what's been set up:

## What Was Created

### 1. Character Reference Document (`kelly_character_reference.md`)
- Complete physical description of Kelly
- Reinmaker armor variant specifications
- Daily Lesson variant specifications  
- Color palette restrictions
- Art style requirements
- Mandatory negative prompts

### 2. Updated Script (`generate_assets.ps1`)
- **Character Reference Variables:** Kelly's base appearance, wardrobe variants, and negative prompts are now hardcoded at the top
- **Build-KellyPrompt Function:** Automatically builds consistent prompts with Kelly's character description
- **Negative Prompt Support:** Script now accepts and includes negative prompts in API calls
- **Updated Asset Prompts:** Player sprite, splash art, and banner now use the character consistency system

## How It Works

### For Any Asset Featuring Kelly:

**Before (Old Way):**
```
"Stylized game sprite of Kelly, the Rein Maker's Daughter..."
```

**After (New Way):**
```
$prompt = Build-KellyPrompt -SceneDescription "..." -WardrobeVariant "Reinmaker" -Pose "..." -Lighting "..."
```

The function automatically:
1. Adds Kelly's full character description
2. Adds the correct wardrobe variant
3. Adds mandatory negative prompts
4. Enforces photorealistic style requirements

## Key Features

### Character Consistency
- **Base Description:** Always includes Kelly's exact facial features, hair, age, build
- **Wardrobe Variants:** Choose "Reinmaker" (armor) or "DailyLesson" (studio) variants
- **Color Palette:** Automatically enforces dark grays, charcoal, steel - NO bright colors

### Style Enforcement
- **Mandatory Negatives:** Automatically excludes cartoons, stylized art, memes, fantasy elements
- **Photorealistic Only:** Forces professional photography quality
- **Realistic Textures:** Ensures realistic skin, fabric, and metallic surfaces

## Usage Examples

### Generate Player Sprite (Reinmaker Armor):
```powershell
$prompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic view of Kelly in neutral running pose" `
    -WardrobeVariant "Reinmaker" `
    -Pose "running pose, dynamic movement" `
    -Lighting "orthographic camera view, soft forge key light" `
    -AdditionalNegatives "stylized, cel-shaded, pixel art"

Generate-Google-Asset -Prompt $prompt.Prompt -NegativePrompt $prompt.Negative ...
```

### Generate Daily Lesson Variant:
```powershell
$prompt = Build-KellyPrompt `
    -SceneDescription "Kelly sitting in director's chair, addressing viewer" `
    -WardrobeVariant "DailyLesson" `
    -Pose "seated, approachable expression" `
    -Lighting "studio lighting, white background"

Generate-Google-Asset -Prompt $prompt.Prompt -NegativePrompt $prompt.Negative ...
```

## Next Steps to Regenerate Assets

1. **Review the character reference:** Open `kelly_character_reference.md` to verify Kelly's appearance matches your vision

2. **Update any prompts:** If needed, modify the character variables at the top of `generate_assets.ps1`

3. **Regenerate assets:** Run `.\generate_assets.ps1` - all Kelly assets will now use consistent character description

4. **For Reference Images:** If you have Kelly's base image file, you can:
   - Save it as `kelly_base_reference.png` 
   - The script can be enhanced to include reference images via base64 encoding if the API supports it

## Important Notes

- **Negative Prompts:** The API may or may not support negative prompts directly. If not, the negative prompts are still included in the prompt text as "avoid: [negatives]"
- **Reference Images:** Currently using text-based character description. For even better consistency, consider using image-to-image generation if available
- **Testing:** Test with one asset first to verify the character consistency before regenerating all assets

## Character Lock-In Checklist

✅ Character base description defined  
✅ Wardrobe variants defined  
✅ Color palette restrictions enforced  
✅ Negative prompts defined  
✅ Prompt building function created  
✅ Key assets updated to use new system  
✅ Documentation created  

Your character consistency system is now locked in and ready to use!












