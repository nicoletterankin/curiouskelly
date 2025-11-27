#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Linq;

namespace KellySetup
{
    public class FixHairAndController : EditorWindow
    {
        [MenuItem("Kelly/🛠️ Fix Hair Position & Setup Controller")]
        public static void Fix()
        {
            GameObject kelly = GameObject.Find("Kelly_Live_v2");
            if (kelly == null)
            {
                kelly = GameObject.Find("Kelly_Live_v1");
            }
            
            if (kelly == null)
            {
                Debug.LogError("Could not find 'Kelly_Live_v2' or 'Kelly_Live_v1' in the scene!");
                return;
            }
            
            Debug.Log($"=== FIXING KELLY ({kelly.name}) ===");
            
            // 1. Fix Hair Position
            // Look for any object with "Hair" in the name
            var hairs = kelly.GetComponentsInChildren<Transform>(true)
                .Where(t => t.name.Contains("Hair") && t.GetComponent<SkinnedMeshRenderer>() != null)
                .ToArray();
                
            if (hairs.Length > 0)
            {
                foreach (var hair in hairs)
                {
                    Undo.RecordObject(hair, "Fix Hair Transform");
                    
                    Debug.Log($"Found Hair: {hair.name}");
                    Debug.Log($"  Old Position: {hair.localPosition}");
                    Debug.Log($"  Old Rotation: {hair.localEulerAngles}");
                    
                    // Reset to zero - this is usually the fix for CC characters
                    hair.localPosition = Vector3.zero;
                    hair.localRotation = Quaternion.identity;
                    hair.localScale = Vector3.one;
                    
                    Debug.Log($"✓ Reset Transform for: {hair.name}");
                }
            }
            else
            {
                Debug.LogWarning("No hair object found! (Looked for 'Hair' with SkinnedMeshRenderer)");
            }
            
            // 2. Setup KellyAvatarController
            var controller = kelly.GetComponent<KellyAvatarController>();
            if (controller != null)
            {
                Undo.RecordObject(controller, "Setup Avatar Controller");
                
                if (controller.faceMesh == null)
                {
                    // Find the face mesh (CC_Base_Body)
                    var body = kelly.GetComponentsInChildren<SkinnedMeshRenderer>()
                        .FirstOrDefault(r => r.name.Contains("CC_Base_Body") || r.name.Contains("Head"));
                        
                    if (body != null)
                    {
                        controller.faceMesh = body;
                        Debug.Log($"✓ Assigned Face Mesh: {body.name}");
                    }
                    else
                    {
                        Debug.LogError("Could not find 'CC_Base_Body' or 'Head' mesh!");
                    }
                }
                else
                {
                    Debug.Log("✓ Face Mesh already assigned.");
                }
            }
            else
            {
                Debug.Log("No KellyAvatarController found on root object.");
            }
            
            Debug.Log("=== FIX COMPLETE ===");
        }
    }
}
#endif

