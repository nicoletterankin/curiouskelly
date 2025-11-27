#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

namespace KellySetup
{
    public class MapARKitBlendshapes : EditorWindow
    {
        // Complete ARKit 52 Blendshape Names
        private static readonly string[] ARKitBlendshapeNames = new string[]
        {
            "browInnerUp", "browDown_L", "browDown_R", "browOuterUp_L", "browOuterUp_R",
            "eyeLookUp_L", "eyeLookUp_R", "eyeLookDown_L", "eyeLookDown_R",
            "eyeLookIn_L", "eyeLookIn_R", "eyeLookOut_L", "eyeLookOut_R",
            "eyeBlink_L", "eyeBlink_R", "eyeSquint_L", "eyeSquint_R",
            "eyeWide_L", "eyeWide_R", "cheekPuff", "cheekSquint_L", "cheekSquint_R",
            "noseSneer_L", "noseSneer_R", "jawOpen", "jawForward", "jawLeft", "jawRight",
            "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight", "mouthRollUpper",
            "mouthRollLower", "mouthShrugUpper", "mouthShrugLower", "mouthClose",
            "mouthSmile_L", "mouthSmile_R", "mouthFrown_L", "mouthFrown_R",
            "mouthDimple_L", "mouthDimple_R", "mouthUpperUp_L", "mouthUpperUp_R",
            "mouthLowerDown_L", "mouthLowerDown_R", "mouthPress_L", "mouthPress_R",
            "mouthStretch_L", "mouthStretch_R", "tongueOut"
        };
        
        [MenuItem("Kelly/4. Map ARKit Blendshapes")]
        public static void MapBlendshapes()
        {
            // Find Kelly prefab
            GameObject kelly = GameObject.Find("Kelly_Live_v2");
            if (kelly == null)
            {
                Debug.LogError("Kelly_Live_v2 not found in scene! Drag Kelly into the scene first.");
                return;
            }
            
            // Find SkinnedMeshRenderer (usually on CC_Base_Body)
            SkinnedMeshRenderer[] renderers = kelly.GetComponentsInChildren<SkinnedMeshRenderer>();
            SkinnedMeshRenderer headRenderer = renderers.FirstOrDefault(r => 
                r.name.Contains("CC_Base_Body") || 
                r.name.Contains("Head") ||
                r.sharedMesh.blendShapeCount > 50
            );
            
            if (headRenderer == null)
            {
                Debug.LogError("Could not find head SkinnedMeshRenderer with blendshapes!");
                return;
            }
            
            Debug.Log($"=== MAPPING ARKIT BLENDSHAPES ON: {headRenderer.name} ===");
            Debug.Log($"Total blendshapes found: {headRenderer.sharedMesh.blendShapeCount}");
            
            // Create mapping dictionary
            Dictionary<string, int> blendshapeMap = new Dictionary<string, int>();
            
            // Map CC blendshapes to ARKit names
            for (int i = 0; i < headRenderer.sharedMesh.blendShapeCount; i++)
            {
                string shapeName = headRenderer.sharedMesh.GetBlendShapeName(i);
                
                // Try to match to ARKit names (CC uses different naming)
                string arkitName = FindARKitMatch(shapeName);
                if (arkitName != null)
                {
                    blendshapeMap[arkitName] = i;
                    Debug.Log($"✓ Mapped: {arkitName} → {shapeName} (index {i})");
                }
            }
            
            Debug.Log($"=== MAPPED {blendshapeMap.Count}/52 ARKIT BLENDSHAPES ===");
            
            // Add ARKitBlendshapeController component
            ARKitBlendshapeController controller = kelly.GetComponent<ARKitBlendshapeController>();
            if (controller == null)
            {
                controller = kelly.AddComponent<ARKitBlendshapeController>();
            }
            
            controller.headRenderer = headRenderer;
            controller.blendshapeMap = blendshapeMap;
            
            Debug.Log("✓ Added ARKitBlendshapeController component");
            
            // Save scene
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(
                UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene()
            );
        }
        
        private static string FindARKitMatch(string ccName)
        {
            // CC naming → ARKit naming (common mappings)
            Dictionary<string, string> nameMap = new Dictionary<string, string>()
            {
                // Eyes
                {"Eye_Blink_L", "eyeBlink_L"},
                {"Eye_Blink_R", "eyeBlink_R"},
                {"Eye_Wide_L", "eyeWide_L"},
                {"Eye_Wide_R", "eyeWide_R"},
                {"Eye_Squint_L", "eyeSquint_L"},
                {"Eye_Squint_R", "eyeSquint_R"},
                
                // Brows
                {"Brow_Inner_Up", "browInnerUp"},
                {"Brow_Down_L", "browDown_L"},
                {"Brow_Down_R", "browDown_R"},
                {"Brow_Outer_Up_L", "browOuterUp_L"},
                {"Brow_Outer_Up_R", "browOuterUp_R"},
                
                // Mouth
                {"Mouth_Smile_L", "mouthSmile_L"},
                {"Mouth_Smile_R", "mouthSmile_R"},
                {"Mouth_Frown_L", "mouthFrown_L"},
                {"Mouth_Frown_R", "mouthFrown_R"},
                {"Jaw_Open", "jawOpen"},
                {"Mouth_Funnel", "mouthFunnel"},
                {"Mouth_Pucker", "mouthPucker"},
                
                // Add more mappings as needed...
            };
            
            return nameMap.ContainsKey(ccName) ? nameMap[ccName] : null;
        }
    }
}
#endif

