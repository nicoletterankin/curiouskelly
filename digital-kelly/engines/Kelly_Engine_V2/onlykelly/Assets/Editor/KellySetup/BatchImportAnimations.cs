#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Linq;

namespace KellySetup
{
    public class BatchImportAnimations : EditorWindow
    {
        private string sourceFolder = "D:/Kelly_Animations/Exports";
        private string destFolder = "Assets/Kelly/Animations/Lessons";
        
        [MenuItem("Kelly/6. Batch Import iClone Animations")]
        public static void ShowWindow()
        {
            GetWindow<BatchImportAnimations>("Import Animations");
        }
        
        private void OnGUI()
        {
            GUILayout.Label("Batch Animation Import", EditorStyles.boldLabel);
            
            sourceFolder = EditorGUILayout.TextField("Source Folder:", sourceFolder);
            destFolder = EditorGUILayout.TextField("Dest Folder:", destFolder);
            
            if (GUILayout.Button("Import All"))
            {
                ImportAll();
            }
        }
        
        private void ImportAll()
        {
            if (!Directory.Exists(sourceFolder))
            {
                Debug.LogError($"Source folder not found: {sourceFolder}");
                return;
            }
            
            Directory.CreateDirectory(destFolder);
            
            string[] fbxFiles = Directory.GetFiles(sourceFolder, "*.fbx");
            Debug.Log($"=== IMPORTING {fbxFiles.Length} ANIMATIONS ===");
            
            int success = 0;
            foreach (string fbxPath in fbxFiles)
            {
                if (ImportAnimation(fbxPath))
                    success++;
            }
            
            AssetDatabase.Refresh();
            Debug.Log($"=== IMPORT COMPLETE: {success}/{fbxFiles.Length} ===");
        }
        
        private bool ImportAnimation(string fbxPath)
        {
            string fileName = Path.GetFileName(fbxPath);
            string destPath = Path.Combine(destFolder, fileName);
            
            try
            {
                // Copy file
                File.Copy(fbxPath, destPath, true);
                
                // Wait for import
                AssetDatabase.ImportAsset(destPath);
                
                // Configure import settings
                ModelImporter importer = AssetImporter.GetAtPath(destPath) as ModelImporter;
                if (importer != null)
                {
                    // Animation-only import
                    importer.importAnimation = true;
                    importer.animationType = ModelImporterAnimationType.Human;
                    importer.importBlendShapes = true;
                    
                    // Don't import mesh/materials (we already have Kelly's base)
                    importer.materialImportMode = ModelImporterMaterialImportMode.None;
                    
                    // Optimize
                    importer.animationCompression = ModelImporterAnimationCompression.KeyframeReduction;
                    importer.animationRotationError = 0.5f;
                    importer.animationPositionError = 0.5f;
                    importer.animationScaleError = 0.5f;
                    
                    EditorUtility.SetDirty(importer);
                    importer.SaveAndReimport();
                    
                    Debug.Log($"✓ Imported: {fileName}");
                    return true;
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"✗ Failed to import {fileName}: {e.Message}");
            }
            
            return false;
        }
    }
}
#endif

