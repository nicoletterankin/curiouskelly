#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using System.IO;

namespace KellySetup
{
    public class CreateSSSProfiles : EditorWindow
    {
        [MenuItem("Kelly/5. Create Subsurface Scattering Profiles")]
        public static void CreateProfiles()
        {
            string profilePath = "Assets/Kelly/Rendering/SSSProfiles";
            if (!Directory.Exists(profilePath))
            {
                Directory.CreateDirectory(profilePath);
            }
            
            Debug.Log("=== CREATING SSS PROFILES ===");
            
            // Profile 1: Caucasian Skin
            CreateSkinProfile(
                profilePath,
                "Kelly_Skin_Caucasian",
                new Color(0.95f, 0.72f, 0.58f), // Base color
                new Vector3(0.48f, 0.41f, 0.28f), // Scatter distance
                0.7f // Scatter intensity
            );
            
            // Profile 2: Asian Skin
            CreateSkinProfile(
                profilePath,
                "Kelly_Skin_Asian",
                new Color(0.98f, 0.82f, 0.65f),
                new Vector3(0.42f, 0.38f, 0.25f),
                0.65f
            );
            
            // Profile 3: African Skin
            CreateSkinProfile(
                profilePath,
                "Kelly_Skin_African",
                new Color(0.45f, 0.32f, 0.22f),
                new Vector3(0.38f, 0.32f, 0.22f),
                0.8f
            );
            
            // Profile 4: Eye Wetness
            CreateEyeProfile(
                profilePath,
                "Kelly_Eye_Wetness"
            );
            
            Debug.Log("=== SSS PROFILES CREATED ===");
            AssetDatabase.Refresh();
        }
        
        private static void CreateSkinProfile(
            string path, 
            string name, 
            Color baseColor, 
            Vector3 scatterDistance,
            float intensity
        )
        {
            // Create ScriptableObject for profile
            // Note: URP doesn't expose DiffusionProfile directly
            // We'll create material overrides instead
            
            Material skinMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            
            // Configure material
            skinMaterial.SetColor("_BaseColor", baseColor);
            skinMaterial.SetFloat("_Smoothness", 0.6f);
            skinMaterial.SetFloat("_Metallic", 0.0f);
            
            // Save material
            string matPath = $"{path}/{name}.mat";
            AssetDatabase.CreateAsset(skinMaterial, matPath);
            Debug.Log($"✓ Created: {matPath}");
        }
        
        private static void CreateEyeProfile(string path, string name)
        {
            Material eyeMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            
            eyeMaterial.SetFloat("_Smoothness", 0.95f);
            eyeMaterial.SetFloat("_Metallic", 0.0f);
            
            string matPath = $"{path}/{name}.mat";
            AssetDatabase.CreateAsset(eyeMaterial, matPath);
            Debug.Log($"✓ Created: {matPath}");
        }
    }
}
#endif

