# Kelly Re-Export Guide (Character Creator 5 → Unity)

## When to Re-Export

Re-export Kelly from Character Creator 5 when:
- After purchasing CC/iC Unity Tools license (removes watermark)
- When updating Kelly's appearance (hair, clothing, skin)
- When adding new facial expressions or morphs
- When fixing issues that require source file changes
- When upgrading to newer CC/iC Unity Tools version

---

## Prerequisites

Before re-exporting, ensure you have:
- [x] Character Creator 5 installed and activated
- [x] Kelly source file (.ccProject or .ccAvatar)
- [x] CC/iC Auto Setup for Unity plugin installed in CC5
- [ ] CC/iC Unity Tools license (optional, for watermark removal)

---

## Step-by-Step Re-Export Process

### Step 1: Open Kelly in Character Creator 5

1. Launch **Character Creator 5**
2. Go to **File > Open Project**
3. Navigate to your Kelly source file
4. Select and open the `.ccProject` file
5. Verify Kelly loads correctly with all clothing/hair

### Step 2: Verify Character Setup

Before exporting, check:
- [ ] Hair is attached correctly
- [ ] Clothing fits properly
- [ ] Skin textures look correct
- [ ] Expression morphs work (test a few)

### Step 3: Configure Export Settings

1. Go to **File > Export > FBX (Clothed Character)**
2. In the Export dialog:

**Target Settings:**
| Setting | Value |
|---------|-------|
| Target | Unity 3D |
| FBX Version | FBX 2020 |

**Mesh Options:**
| Setting | Value |
|---------|-------|
| Delete Hidden Faces | Unchecked |
| Merge Face Hair to One Object | Unchecked |
| Subdivide | None |

**Motion Options:**
| Setting | Value |
|---------|-------|
| Current Pose | Selected |
| Include Motion | Unchecked (unless exporting animations) |

**Texture Options:**
| Setting | Value |
|---------|-------|
| Embed Textures | Unchecked |
| Texture Size | Original |
| Convert to PNG | Checked |

**Advanced:**
| Setting | Value |
|---------|-------|
| Export with CC/iC Auto Setup | Checked (if licensed) |

### Step 4: Export

1. Set export filename: `kelly_fbx_v5.fbx` (increment version)
2. Choose export location: `C:\Kelly_Animations\Exports\`
3. Click **Export**
4. Wait for export to complete (1-2 minutes)

### Step 5: Verify Export

Check the export folder contains:
```
kelly_fbx_v5.fbx           (main model file)
kelly_fbx_v5.fbm/          (textures folder)
kelly_fbx_v5.json          (metadata)
```

---

## Import to Unity

### Step 1: Copy Files

1. Copy `kelly_fbx_v5.fbx` to Unity project:
   ```
   digital-kelly/engines/Kelly_Engine_V2/onlykelly/Assets/
   ```
2. Copy the `.fbm` textures folder alongside it

### Step 2: Unity Auto-Import

1. Switch to Unity (it will detect new files)
2. Wait for import progress bar to complete
3. Check Console for any errors

### Step 3: Configure Import Settings

1. Select `kelly_fbx_v5` in Project panel
2. In Inspector, set:

**Model Tab:**
| Setting | Value |
|---------|-------|
| Scale Factor | 1 |
| Import BlendShapes | Checked |
| Import Visibility | Checked |
| Import Cameras | Unchecked |
| Import Lights | Unchecked |

**Rig Tab:**
| Setting | Value |
|---------|-------|
| Animation Type | Humanoid |
| Avatar Definition | Create From This Model |

**Materials Tab:**
| Setting | Value |
|---------|-------|
| Material Creation Mode | Import via MaterialDescription |
| Location | Use Embedded Materials |

3. Click **Apply**

### Step 4: Replace in Scene

1. Open `Assets/Scenes/KellyMain.unity`
2. Delete old Kelly from Hierarchy
3. Drag new `kelly_fbx_v5` into Hierarchy
4. Set Transform:
   - Position: (0, 0, 0)
   - Rotation: (0, 0, 0)
   - Scale: (1, 1, 1)
5. Save scene (Ctrl+S)

### Step 5: Fix Materials (if needed)

If hair appears transparent:
1. Find hair material in Project panel
2. Change Surface Type: Transparent → Opaque
3. Enable Alpha Clipping, Threshold: 0.5

### Step 6: Rebuild and Deploy

1. **Kelly > Build > 🚀 Build WebGL (Production)**
2. Wait for build (20-30 minutes)
3. Run: `.\deploy-kelly.ps1`
4. Verify at Netlify URL

---

## Exporting Animations from iClone

### For Lip Sync Animations:

1. Open Kelly in **iClone 8**
2. Add audio track for lesson
3. Use **AccuLips** to generate lip sync
4. **File > Export > Export FBX**
5. Settings:
   - Target: Unity 3D
   - Include: Animation Only
   - BlendShapes: Checked
6. Import to Unity as Animation Clip
7. Add to Animator Controller

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| kelly_fbx_v1 | Initial | First export |
| kelly_fbx_v2 | - | Hair fixes |
| kelly_fbx_v3 | - | Material updates |
| kelly_fbx_v4 | Nov 26, 2025 | Current production |
| kelly_fbx_v5 | TBD | Post-license export |

---

## Troubleshooting

### "Export failed" in CC5
- Ensure enough disk space
- Try exporting to different location
- Restart CC5 and try again

### Materials look wrong in Unity
- Re-run CC/iC Auto Setup
- Check texture paths are correct
- Verify .fbm folder was copied

### Blendshapes missing
- Ensure "Import BlendShapes" is checked
- Re-export with "Include Morphs" enabled

---

*Last Updated: November 26, 2025*

