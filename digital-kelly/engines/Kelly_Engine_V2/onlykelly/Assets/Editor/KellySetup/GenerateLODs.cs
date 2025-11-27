#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

namespace KellySetup
{
    public class GenerateLODs : EditorWindow
    {
        [MenuItem("Kelly/7. Generate LOD Groups")]
        public static void Generate()
        {
            GameObject kelly = GameObject.Find("Kelly_Live_v2");
            if (kelly == null)
            {
                Debug.LogError("Kelly not found in scene!");
                return;
            }
            
            Debug.Log("=== GENERATING LOD GROUPS ===");
            
            // Get all SkinnedMeshRenderers
            SkinnedMeshRenderer[] renderers = kelly.GetComponentsInChildren<SkinnedMeshRenderer>();
            
            foreach (var renderer in renderers)
            {
                GenerateLODForMesh(renderer);
            }
            
            Debug.Log("=== LOD GENERATION COMPLETE ===");
        }
        
        private static void GenerateLODForMesh(SkinnedMeshRenderer originalRenderer)
        {
            GameObject parent = originalRenderer.transform.parent.gameObject;
            
            // Add LODGroup component
            LODGroup lodGroup = parent.GetComponent<LODGroup>();
            if (lodGroup == null)
            {
                lodGroup = parent.AddComponent<LODGroup>();
            }
            
            // Create LOD levels
            LOD[] lods = new LOD[4];
            
            // LOD 0 - Original (100% quality, 0-60% screen height)
            lods[0] = new LOD(0.6f, new Renderer[] { originalRenderer });
            
            // LOD 1 - 50% reduction (60-30% screen height)
            SkinnedMeshRenderer lod1 = CreateReducedMesh(originalRenderer, 0.5f, "LOD1");
            lods[1] = new LOD(0.3f, new Renderer[] { lod1 });
            
            // LOD 2 - 75% reduction (30-15% screen height)
            SkinnedMeshRenderer lod2 = CreateReducedMesh(originalRenderer, 0.25f, "LOD2");
            lods[2] = new LOD(0.15f, new Renderer[] { lod2 });
            
            // LOD 3 - Culled (15-0% screen height)
            lods[3] = new LOD(0f, new Renderer[] { });
            
            lodGroup.SetLODs(lods);
            lodGroup.RecalculateBounds();
            
            Debug.Log($"✓ Created LOD group for: {originalRenderer.name}");
        }
        
        private static SkinnedMeshRenderer CreateReducedMesh(
            SkinnedMeshRenderer original, 
            float quality, 
            string suffix
        )
        {
            // Duplicate renderer
            GameObject lodObj = new GameObject(original.name + "_" + suffix);
            lodObj.transform.SetParent(original.transform.parent);
            lodObj.transform.localPosition = Vector3.zero;
            lodObj.transform.localRotation = Quaternion.identity;
            lodObj.transform.localScale = Vector3.one;
            
            SkinnedMeshRenderer lodRenderer = lodObj.AddComponent<SkinnedMeshRenderer>();
            
            // Simplify mesh
            Mesh simplifiedMesh = SimplifyMesh(original.sharedMesh, quality);
            lodRenderer.sharedMesh = simplifiedMesh;
            
            // Copy properties
            lodRenderer.sharedMaterials = original.sharedMaterials;
            lodRenderer.bones = original.bones;
            lodRenderer.rootBone = original.rootBone;
            lodRenderer.quality = SkinQuality.Auto;
            
            // Disable by default (LODGroup will control)
            lodRenderer.enabled = false;
            
            return lodRenderer;
        }
        
        private static Mesh SimplifyMesh(Mesh original, float quality)
        {
            // Unity doesn't have built-in mesh simplification
            // Options:
            // 1. Use UnityMeshSimplifier package (GitHub)
            // 2. Use InstaLOD plugin
            // 3. Pre-generate in Blender/Maya
            
            // For now, return a copy (you'll need to integrate a simplifier)
            Mesh simplified = Object.Instantiate(original);
            simplified.name = original.name + "_Simplified";
            
            // TODO: Integrate mesh simplification library
            // Example with UnityMeshSimplifier:
            // var simplifier = new UnityMeshSimplifier.MeshSimplifier();
            // simplifier.Initialize(original);
            // simplifier.SimplifyMesh(quality);
            // simplified = simplifier.ToMesh();
            
            return simplified;
        }
    }
}
#endif

