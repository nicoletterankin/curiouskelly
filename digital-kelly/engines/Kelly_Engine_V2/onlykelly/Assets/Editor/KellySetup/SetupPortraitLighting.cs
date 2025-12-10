#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace KellySetup
{
    /// <summary>
    /// Sets up professional 3-point portrait lighting for Kelly
    /// Run via: Kelly > Lighting > Setup Portrait Lighting
    /// </summary>
    public class SetupPortraitLighting : EditorWindow
    {
        [MenuItem("Kelly/Lighting/💡 Setup Portrait Lighting", false, 200)]
        public static void SetupLighting()
        {
            Debug.Log("╔══════════════════════════════════════════════════════╗");
            Debug.Log("║         KELLY PORTRAIT LIGHTING SETUP                ║");
            Debug.Log("╚══════════════════════════════════════════════════════╝");

            // Find Kelly in the scene
            GameObject kelly = FindKelly();
            if (kelly == null)
            {
                Debug.LogError("✗ Could not find Kelly in the scene!");
                Debug.LogError("  Make sure Kelly is in the hierarchy before running this.");
                return;
            }

            Vector3 kellyPosition = kelly.transform.position;
            float kellyHeight = 1.6f; // Approximate head height

            // Create lighting parent
            GameObject lightingRig = new GameObject("Kelly_LightingRig");
            Undo.RegisterCreatedObjectUndo(lightingRig, "Create Lighting Rig");
            lightingRig.transform.position = kellyPosition;

            // 1. RIM LIGHT (Back light for hair separation)
            Debug.Log("▶ Adding Rim Light...");
            GameObject rimLightObj = new GameObject("Kelly_RimLight");
            rimLightObj.transform.parent = lightingRig.transform;
            rimLightObj.transform.position = kellyPosition + new Vector3(0, kellyHeight + 0.5f, -2f);
            rimLightObj.transform.LookAt(kellyPosition + Vector3.up * kellyHeight);
            
            Light rimLight = rimLightObj.AddComponent<Light>();
            rimLight.type = LightType.Directional;
            rimLight.intensity = 0.3f;
            rimLight.color = new Color(1f, 0.95f, 0.9f); // Warm white
            rimLight.shadows = LightShadows.None;
            rimLight.renderMode = LightRenderMode.ForcePixel;
            Debug.Log("  ✓ Rim Light: Directional, 0.3 intensity, behind Kelly");

            // 2. FILL LIGHT (Opposite side of key light)
            Debug.Log("▶ Adding Fill Light...");
            GameObject fillLightObj = new GameObject("Kelly_FillLight");
            fillLightObj.transform.parent = lightingRig.transform;
            fillLightObj.transform.position = kellyPosition + new Vector3(-1.5f, kellyHeight, 1.5f);
            fillLightObj.transform.LookAt(kellyPosition + Vector3.up * kellyHeight);
            
            Light fillLight = fillLightObj.AddComponent<Light>();
            fillLight.type = LightType.Point;
            fillLight.intensity = 0.2f;
            fillLight.range = 5f;
            fillLight.color = new Color(0.9f, 0.95f, 1f); // Cool white (opposite of warm key)
            fillLight.shadows = LightShadows.None;
            fillLight.renderMode = LightRenderMode.ForcePixel;
            Debug.Log("  ✓ Fill Light: Point, 0.2 intensity, camera-left");

            // 3. EYE CATCH LIGHTS (Small spots for eye reflections)
            Debug.Log("▶ Adding Eye Catch Lights...");
            
            // Left eye catch light
            GameObject eyeCatchLeftObj = new GameObject("Kelly_EyeCatchLight_L");
            eyeCatchLeftObj.transform.parent = lightingRig.transform;
            eyeCatchLeftObj.transform.position = kellyPosition + new Vector3(-0.3f, kellyHeight + 0.3f, 1f);
            eyeCatchLeftObj.transform.LookAt(kellyPosition + new Vector3(-0.05f, kellyHeight, 0));
            
            Light eyeCatchLeft = eyeCatchLeftObj.AddComponent<Light>();
            eyeCatchLeft.type = LightType.Spot;
            eyeCatchLeft.spotAngle = 15f;
            eyeCatchLeft.intensity = 0.5f;
            eyeCatchLeft.range = 3f;
            eyeCatchLeft.color = Color.white;
            eyeCatchLeft.shadows = LightShadows.None;
            eyeCatchLeft.renderMode = LightRenderMode.ForcePixel;

            // Right eye catch light
            GameObject eyeCatchRightObj = new GameObject("Kelly_EyeCatchLight_R");
            eyeCatchRightObj.transform.parent = lightingRig.transform;
            eyeCatchRightObj.transform.position = kellyPosition + new Vector3(0.3f, kellyHeight + 0.3f, 1f);
            eyeCatchRightObj.transform.LookAt(kellyPosition + new Vector3(0.05f, kellyHeight, 0));
            
            Light eyeCatchRight = eyeCatchRightObj.AddComponent<Light>();
            eyeCatchRight.type = LightType.Spot;
            eyeCatchRight.spotAngle = 15f;
            eyeCatchRight.intensity = 0.5f;
            eyeCatchRight.range = 3f;
            eyeCatchRight.color = Color.white;
            eyeCatchRight.shadows = LightShadows.None;
            eyeCatchRight.renderMode = LightRenderMode.ForcePixel;
            
            Debug.Log("  ✓ Eye Catch Lights: 2x Spot lights aimed at eyes");

            // Mark scene dirty
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(
                UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene()
            );

            Debug.Log("");
            Debug.Log("╔══════════════════════════════════════════════════════╗");
            Debug.Log("║              LIGHTING SETUP COMPLETE                  ║");
            Debug.Log("╚══════════════════════════════════════════════════════╝");
            Debug.Log("");
            Debug.Log("✓ Created lighting rig with:");
            Debug.Log("  • Rim Light (behind, 0.3 intensity)");
            Debug.Log("  • Fill Light (left, 0.2 intensity)");
            Debug.Log("  • Eye Catch Lights (2x spots for eye reflections)");
            Debug.Log("");
            Debug.Log("NEXT STEPS:");
            Debug.Log("1. Adjust light positions if needed (select Kelly_LightingRig)");
            Debug.Log("2. Test in Play mode");
            Debug.Log("3. Save scene (Ctrl+S)");
            Debug.Log("4. Build WebGL to test changes");
        }

        private static GameObject FindKelly()
        {
            // Try common names
            string[] possibleNames = new string[]
            {
                "Kelly",
                "kelly",
                "Kelly_Live_v2",
                "Kelly_Live",
                "CC_Base_Body",
                "kelly_fbx_v4"
            };

            foreach (string name in possibleNames)
            {
                GameObject obj = GameObject.Find(name);
                if (obj != null) return obj;
            }

            // Search for any object with "Kelly" in the name
            GameObject[] allObjects = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None);
            foreach (GameObject obj in allObjects)
            {
                if (obj.name.ToLower().Contains("kelly"))
                {
                    return obj;
                }
            }

            // Search for SkinnedMeshRenderer (likely Kelly's body)
            SkinnedMeshRenderer[] renderers = Object.FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None);
            foreach (SkinnedMeshRenderer renderer in renderers)
            {
                if (renderer.sharedMesh != null && 
                    (renderer.sharedMesh.name.ToLower().Contains("body") ||
                     renderer.sharedMesh.name.ToLower().Contains("head")))
                {
                    return renderer.gameObject.transform.root.gameObject;
                }
            }

            return null;
        }

        [MenuItem("Kelly/Lighting/🔧 Adjust Lighting Intensity", false, 201)]
        public static void ShowLightingAdjuster()
        {
            GetWindow<SetupPortraitLighting>("Lighting Adjuster");
        }

        private float rimIntensity = 0.3f;
        private float fillIntensity = 0.2f;
        private float eyeCatchIntensity = 0.5f;
        private Color rimColor = new Color(1f, 0.95f, 0.9f);
        private Color fillColor = new Color(0.9f, 0.95f, 1f);

        void OnGUI()
        {
            GUILayout.Label("Kelly Lighting Adjuster", EditorStyles.boldLabel);
            GUILayout.Space(10);

            GUILayout.Label("Rim Light (Hair Separation)", EditorStyles.miniBoldLabel);
            rimIntensity = EditorGUILayout.Slider("Intensity", rimIntensity, 0f, 1f);
            rimColor = EditorGUILayout.ColorField("Color", rimColor);
            GUILayout.Space(5);

            GUILayout.Label("Fill Light (Shadow Fill)", EditorStyles.miniBoldLabel);
            fillIntensity = EditorGUILayout.Slider("Intensity", fillIntensity, 0f, 1f);
            fillColor = EditorGUILayout.ColorField("Color", fillColor);
            GUILayout.Space(5);

            GUILayout.Label("Eye Catch Lights", EditorStyles.miniBoldLabel);
            eyeCatchIntensity = EditorGUILayout.Slider("Intensity", eyeCatchIntensity, 0f, 2f);
            GUILayout.Space(10);

            if (GUILayout.Button("Apply Changes"))
            {
                ApplyLightingChanges();
            }
        }

        private void ApplyLightingChanges()
        {
            Light rimLight = FindLightByName("Kelly_RimLight");
            if (rimLight != null)
            {
                rimLight.intensity = rimIntensity;
                rimLight.color = rimColor;
            }

            Light fillLight = FindLightByName("Kelly_FillLight");
            if (fillLight != null)
            {
                fillLight.intensity = fillIntensity;
                fillLight.color = fillColor;
            }

            Light eyeCatchL = FindLightByName("Kelly_EyeCatchLight_L");
            if (eyeCatchL != null)
            {
                eyeCatchL.intensity = eyeCatchIntensity;
            }

            Light eyeCatchR = FindLightByName("Kelly_EyeCatchLight_R");
            if (eyeCatchR != null)
            {
                eyeCatchR.intensity = eyeCatchIntensity;
            }

            Debug.Log("✓ Lighting changes applied!");
        }

        private Light FindLightByName(string name)
        {
            GameObject obj = GameObject.Find(name);
            if (obj != null)
            {
                return obj.GetComponent<Light>();
            }
            return null;
        }
    }
}
#endif





