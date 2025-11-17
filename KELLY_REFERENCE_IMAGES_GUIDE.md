# Kelly Reference Images Guide - Where to Put New Photos

## 📍 **PRIMARY LOCATION FOR REFERENCE IMAGES**

**Put ALL new Kelly reference photos here:**
```
iLearnStudio/projects/Kelly/Ref/
```

## 🎯 **What Makes a Good Reference Image?**

### ✅ **Ideal Reference Images:**
1. **High Quality:** 2K+ resolution (preferably 4K-8K)
2. **Clear Face:** Full face visible, good lighting
3. **Consistent Style:** Photorealistic, matches your desired output style
4. **Clean Background:** White/neutral background preferred
5. **Different Angles:** Front, profile, three-quarter views
6. **Director's Chair Context:** Images of Kelly sitting in the chair (for chair positioning reference)

### 📸 **Recommended Reference Image Types:**

#### **Character Consistency (Face & Hair)**
- `kelly_front.png` - Front-facing, full face visible
- `kelly_profile.png` - Side profile showing facial structure
- `kelly_three_quarter.png` - 45-degree angle view
- `kelly_closeup.png` - Close-up of face (optional but helpful)

#### **Chair Positioning Reference** ⭐ **NEW - CRITICAL**
- `kelly_chair_reference.png` - Kelly sitting properly in director's chair
- `kelly_chair_side.png` - Side view showing chair positioning
- `kelly_chair_front.png` - Front view with chair visible

**Why these matter:** The AI needs to see how Kelly sits in the chair to avoid the "chair floating in front" issue.

## 📁 **Current Reference Image Structure**

```
iLearnStudio/projects/Kelly/Ref/
├── kelly_front.png              ✅ Primary front reference
├── kelly_profile.png            ✅ Profile reference  
├── kelly_three_quarter.png      ✅ Three-quarter reference
├── kelly_chair_reference.png    ⭐ ADD THIS - Chair positioning
└── README.md                    📖 This guide
```

## 🚀 **How to Add New Reference Images**

### **Step 1: Copy Your Photos**
1. Copy your Kelly photos to: `iLearnStudio/projects/Kelly/Ref/`
2. Use descriptive names:
   - `kelly_chair_reference.png` (for chair positioning)
   - `kelly_front_v2.png` (if updating front reference)
   - `kelly_hair_detail.png` (for hair consistency)

### **Step 2: Verify Image Quality**
- ✅ High resolution (at least 1920x1080, preferably 4K+)
- ✅ Clear, sharp focus on Kelly's face
- ✅ Good lighting (not too dark or overexposed)
- ✅ Photorealistic style (matches your desired output)

### **Step 3: Test Generation**
After adding new reference images, run:
```bash
python scripts/generate_kelly_expressions.py
```

The script will automatically detect and use all reference images in the `Ref/` directory.

## 🔧 **How Reference Images Are Used**

### **Current Process:**
1. Script scans `iLearnStudio/projects/Kelly/Ref/` for all `.png` and `.jpg` files
2. Loads reference images using Vertex AI SDK
3. Passes them to the image generation API
4. Uses them for character consistency (face, hair, features)

### **What Gets Improved:**
- ✅ **Face Consistency:** Same face shape, features across all generations
- ✅ **Hair Consistency:** Same hair length, style, color
- ✅ **Chair Positioning:** Proper sitting position (when chair reference provided)
- ✅ **Overall Likeness:** Kelly looks like Kelly in every image

## ⚠️ **Common Issues & Solutions**

### **Issue: "Chair floating in front of Kelly"**
**Solution:** Add a chair reference image showing Kelly properly seated
- Put it in: `iLearnStudio/projects/Kelly/Ref/kelly_chair_reference.png`
- Should show Kelly sitting IN the chair, not in front of it

### **Issue: "Face looks different in each image"**
**Solution:** Add more reference images from different angles
- Front, profile, and three-quarter views help the AI understand the 3D structure

### **Issue: "Hair looks inconsistent"**
**Solution:** Add a hair detail reference image
- Close-up showing hair texture, length, and style
- Put it in: `iLearnStudio/projects/Kelly/Ref/kelly_hair_detail.png`

## 📊 **Best Practices**

1. **Start with 3-5 reference images** - More isn't always better
2. **Use your BEST quality images** - Higher quality = better consistency
3. **Keep naming consistent** - Use `kelly_` prefix and descriptive names
4. **Update references when needed** - If Kelly's appearance changes, update references
5. **Test after adding** - Always test generation after adding new references

## 🎯 **Next Steps**

1. ✅ **Add chair reference image** - Fix the "chair floating" issue
2. ✅ **Add more angle views** - Improve 3D consistency
3. ✅ **Update generation script** - Ensure it uses all references properly
4. ✅ **Test and iterate** - Generate test images and refine

---

**Last Updated:** Generated automatically  
**Questions?** Check `scripts/generate_kelly_expressions.py` for implementation details




