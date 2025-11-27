#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace KellySetup
{
    public class ConfigureProject : EditorWindow
    {
        [MenuItem("Kelly/1. Configure Project Settings")]
        public static void Configure()
        {
            Debug.Log("=== KELLY PROJECT CONFIGURATION START ===");
            
            // 1. Set Color Space to Linear
            PlayerSettings.colorSpace = ColorSpace.Linear;
            Debug.Log("✓ Color Space: Linear");
            
            // 2. API Compatibility Level
            PlayerSettings.SetApiCompatibilityLevel(
                BuildTargetGroup.WebGL, 
                ApiCompatibilityLevel.NET_Standard
            );
            Debug.Log("✓ API Compatibility: .NET Standard");
            
            // 3. WebGL Settings
            PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Brotli;
            PlayerSettings.WebGL.memorySize = 2048; // 2GB
            PlayerSettings.WebGL.exceptionSupport = WebGLExceptionSupport.None;
            PlayerSettings.WebGL.dataCaching = true;
            PlayerSettings.stripEngineCode = true;
            Debug.Log("✓ WebGL: Brotli, 2GB memory, data caching enabled");
            
            // 4. Graphics Settings
            PlayerSettings.SetGraphicsAPIs(BuildTarget.WebGL, new GraphicsDeviceType[] { 
                GraphicsDeviceType.OpenGLES3 
            });
            Debug.Log("✓ Graphics API: OpenGL ES 3.0");
            
            // 5. URP Asset Configuration
            ConfigureURPAsset();
            
            // 6. Quality Settings
            QualitySettings.shadowDistance = 50f;
            QualitySettings.shadowResolution = UnityEngine.ShadowResolution.High;
            QualitySettings.skinWeights = SkinWeights.FourBones;
            QualitySettings.anisotropicFiltering = AnisotropicFiltering.ForceEnable;
            Debug.Log("✓ Quality Settings optimized");
            
            // 7. Physics Settings (disable if not needed)
            Physics.simulationMode = SimulationMode.Script;
            Debug.Log("✓ Physics disabled (not needed for Kelly)");
            
            AssetDatabase.SaveAssets();
            Debug.Log("=== KELLY PROJECT CONFIGURATION COMPLETE ===");
        }
        
        private static void ConfigureURPAsset()
        {
            // Find URP Asset
            var urpAssets = AssetDatabase.FindAssets("t:UniversalRenderPipelineAsset");
            if (urpAssets.Length == 0)
            {
                Debug.LogError("No URP Asset found! Create one first.");
                return;
            }
            
            var urpAssetPath = AssetDatabase.GUIDToAssetPath(urpAssets[0]);
            var urpAsset = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(urpAssetPath);
            
            // Configure URP via SerializedObject (because properties are internal)
            SerializedObject serializedAsset = new SerializedObject(urpAsset);
            
            // Enable HDR
            serializedAsset.FindProperty("m_SupportsHDR").boolValue = true;
            
            // MSAA
            serializedAsset.FindProperty("m_MSAA").intValue = 4; // 4x MSAA
            
            // Shadows
            serializedAsset.FindProperty("m_MainLightShadowmapResolution").intValue = 2048;
            serializedAsset.FindProperty("m_AdditionalLightsShadowmapResolution").intValue = 512;
            
            serializedAsset.ApplyModifiedProperties();
            Debug.Log($"✓ URP Asset configured: {urpAssetPath}");
        }
    }
}
#endif

