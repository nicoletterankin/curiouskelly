#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Linq;

namespace KellySetup
{
    public class ConfigureKellyAvatar : EditorWindow
    {
        [MenuItem("Kelly/3. Configure Kelly Avatar (Fix 'No Avatar')")]
        public static void Configure()
        {
            // Find Kelly FBX
            string[] guids = AssetDatabase.FindAssets("Kelly_Live t:Model");
            if (guids.Length == 0)
            {
                Debug.LogError("Kelly_Live FBX not found!");
                return;
            }
            
            string path = AssetDatabase.GUIDToAssetPath(guids[0]);
            Debug.Log($"Found Kelly at: {path}");
            
            // Get Model Importer
            ModelImporter importer = AssetImporter.GetAtPath(path) as ModelImporter;
            if (importer == null)
            {
                Debug.LogError("Failed to get ModelImporter");
                return;
            }
            
            Debug.Log("=== CONFIGURING KELLY AVATAR ===");
            
            // 1. Animation Type: Humanoid
            importer.animationType = ModelImporterAnimationType.Human;
            Debug.Log("✓ Animation Type: Humanoid");
            
            // 2. Avatar Definition: Create From This Model
            importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            Debug.Log("✓ Avatar Definition: Create From This Model");
            
            // 3. Skin Weights: 4 Bones
            importer.skinWeights = ModelImporterSkinWeights.Standard; // 4 bones
            Debug.Log("✓ Skin Weights: 4 Bones");
            
            // 4. Optimize Game Objects: OFF (we need full hierarchy for blendshapes)
            importer.optimizeGameObjects = false;
            Debug.Log("✓ Optimize Game Objects: OFF");
            
            // 5. Import BlendShapes: ON
            importer.importBlendShapes = true;
            Debug.Log("✓ Import BlendShapes: ON");
            
            // 6. Import Visibility: ON
            importer.importVisibility = true;
            Debug.Log("✓ Import Visibility: ON");
            
            // 7. Import Cameras & Lights: OFF
            importer.importCameras = false;
            importer.importLights = false;
            Debug.Log("✓ Import Cameras/Lights: OFF");
            
            // 8. Mesh Compression: OFF (quality over size for Kelly)
            importer.meshCompression = ModelImporterMeshCompression.Off;
            Debug.Log("✓ Mesh Compression: OFF");
            
            // 9. Read/Write: ON (needed for LOD generation)
            importer.isReadable = true;
            Debug.Log("✓ Read/Write: ON");
            
            // 10. Generate Colliders: OFF
            importer.addCollider = false;
            Debug.Log("✓ Generate Colliders: OFF");
            
            // 11. Normals: Import
            importer.importNormals = ModelImporterNormals.Import;
            importer.normalCalculationMode = ModelImporterNormalCalculationMode.AreaAndAngleWeighted;
            Debug.Log("✓ Normals: Import (Area & Angle Weighted)");
            
            // 12. Tangents: Calculate Mikktspace
            importer.importTangents = ModelImporterTangents.CalculateMikk;
            Debug.Log("✓ Tangents: Calculate Mikktspace");
            
            // 13. Smoothing Angle
            importer.normalSmoothingAngle = 60f;
            Debug.Log("✓ Smoothing Angle: 60°");
            
            // Apply and Reimport
            EditorUtility.SetDirty(importer);
            importer.SaveAndReimport();
            
            Debug.Log("=== AVATAR CONFIGURATION COMPLETE ===");
            Debug.Log("Kelly now has a proper Humanoid Avatar!");
            
            // Auto-select Kelly in Project window
            Selection.activeObject = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        }
    }
}
#endif

