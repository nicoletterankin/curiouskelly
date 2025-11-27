#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections;

namespace KellySetup
{
    public class MasterSetup : EditorWindow
    {
        [MenuItem("Kelly/🚀 MASTER SETUP - RUN ALL")]
        public static void RunAll()
        {
            Debug.Log("╔══════════════════════════════════════════════╗");
            Debug.Log("║   KELLY MASTER SETUP - FULL AUTOMATION      ║");
            Debug.Log("╚══════════════════════════════════════════════╝");
            Debug.Log("");
            
            // Step 1: Project Configuration
            Debug.Log("► STEP 1: Configuring Project Settings...");
            ConfigureProject.Configure();
            
            // Step 2: Install Packages
            Debug.Log("► STEP 2: Installing Required Packages...");
            InstallPackages.Install();
            
            // Wait for package installation
            EditorApplication.delayCall += () =>
            {
                // Step 3: Configure Avatar
                Debug.Log("► STEP 3: Configuring Kelly Avatar...");
                ConfigureKellyAvatar.Configure();
                
                // Step 4: Map Blendshapes
                Debug.Log("► STEP 4: Mapping ARKit Blendshapes...");
                MapARKitBlendshapes.MapBlendshapes();
                
                // Step 5: Create SSS Profiles
                Debug.Log("► STEP 5: Creating SSS Profiles...");
                CreateSSSProfiles.CreateProfiles();
                
                // Step 6: Generate LODs
                Debug.Log("► STEP 6: Generating LOD Groups...");
                GenerateLODs.Generate();
                
                // Step 7: Import Animations
                Debug.Log("► STEP 7: Ready to import animations");
                Debug.Log("   Run: Kelly > 6. Batch Import iClone Animations");
                
                // Done
                Debug.Log("");
                Debug.Log("╔══════════════════════════════════════════════╗");
                Debug.Log("║           MASTER SETUP COMPLETE              ║");
                Debug.Log("║                                              ║");
                Debug.Log("║  Kelly is now fully configured and ready!    ║");
                Debug.Log("║                                              ║");
                Debug.Log("║  Next steps:                                 ║");
                Debug.Log("║  1. Import iClone animations                 ║");
                Debug.Log("║  2. Test in Play mode                        ║");
                Debug.Log("║  3. Build WebGL                              ║");
                Debug.Log("╚══════════════════════════════════════════════╝");
            };
        }
    }
}
#endif

