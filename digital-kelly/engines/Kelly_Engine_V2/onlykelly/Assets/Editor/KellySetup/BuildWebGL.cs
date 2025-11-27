#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using System;
using System.IO;
using System.Linq;

namespace KellySetup
{
    /// <summary>
    /// Kelly V2 WebGL Build System - Zero Hardcoded Paths Edition
    /// Automatically detects the active scene and builds with zero assumptions
    /// </summary>
    public class BuildWebGL
    {
        private const string SCENES_FOLDER = "Assets/Scenes";
        private const string PREFERRED_SCENE_NAME = "KellyMain.unity";
        private const string OUTPUT_PATH = "Builds/WebGL";

        [MenuItem("Kelly/Build/🚀 Build WebGL (Production)", false, 900)]
        public static void BuildProduction()
        {
            BuildWebGLInternal(false);
        }

        [MenuItem("Kelly/Build/🔧 Build WebGL (Development)", false, 901)]
        public static void BuildDevelopment()
        {
            BuildWebGLInternal(true);
        }

        private static void BuildWebGLInternal(bool isDevelopment)
        {
            Debug.Log("╔══════════════════════════════════════════════════════╗");
            Debug.Log("║          KELLY V2 WEBGL BUILD SYSTEM                 ║");
            Debug.Log("╚══════════════════════════════════════════════════════╝");
            Debug.Log("");

            DateTime startTime = DateTime.Now;

            // STEP 1: Detect the scene to build
            string scenePath = DetectSceneToBuild();
            if (string.IsNullOrEmpty(scenePath))
            {
                Debug.LogError("✗ FATAL ERROR: No valid scene found in project!");
                Debug.LogError("  Expected location: Assets/Scenes/");
                Debug.LogError("  Please create a scene or check your Scenes folder.");
                return;
            }

            Debug.Log($"✓ Scene detected: {scenePath}");
            Debug.Log("");

            // STEP 2: Configure build options
            BuildPlayerOptions buildOptions = new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = OUTPUT_PATH,
                target = BuildTarget.WebGL,
                options = isDevelopment ? 
                    BuildOptions.Development | BuildOptions.AllowDebugging : 
                    BuildOptions.None
            };

            // STEP 3: Configure WebGL specific settings
            ConfigureWebGLSettings(isDevelopment);

            // STEP 4: Execute build
            Debug.Log("▼ BUILDING WEBGL...");
            Debug.Log($"  Mode: {(isDevelopment ? "DEVELOPMENT" : "PRODUCTION")}");
            Debug.Log($"  Output: {OUTPUT_PATH}");
            Debug.Log("");

            BuildReport report = BuildPipeline.BuildPlayer(buildOptions);
            BuildSummary summary = report.summary;

            // STEP 5: Report results
            Debug.Log("");
            Debug.Log("╔══════════════════════════════════════════════════════╗");
            Debug.Log("║                  BUILD COMPLETE                       ║");
            Debug.Log("╚══════════════════════════════════════════════════════╝");
            Debug.Log("");

            TimeSpan duration = DateTime.Now - startTime;

            if (summary.result == BuildResult.Succeeded)
            {
                Debug.Log($"✓ BUILD SUCCEEDED");
                Debug.Log($"  Build size: {FormatBytes(summary.totalSize)}");
                Debug.Log($"  Build time: {duration.TotalSeconds:F1} seconds");
                Debug.Log($"  Output path: {Path.GetFullPath(OUTPUT_PATH)}");
                Debug.Log($"  Scene used: {scenePath}");
                Debug.Log("");
                Debug.Log("NEXT STEPS:");
                Debug.Log("1. Test locally: python -m http.server 8000 (in Builds/WebGL)");
                Debug.Log("2. Deploy to Vercel or Netlify");
                Debug.Log("");
            }
            else
            {
                Debug.LogError($"✗ BUILD FAILED: {summary.result}");
                Debug.LogError($"  Total errors: {summary.totalErrors}");
                Debug.LogError($"  Total warnings: {summary.totalWarnings}");
                Debug.LogError($"  Build time: {duration.TotalSeconds:F1} seconds");
                Debug.LogError("");
                Debug.LogError("Check the Console for detailed error messages.");
            }
        }

        /// <summary>
        /// Automatically detects which scene to build
        /// Priority: 
        /// 1. Currently open scene (if saved)
        /// 2. KellyMain.unity (preferred canonical scene)
        /// 3. Any scene in Assets/Scenes/
        /// 4. First scene in build settings
        /// </summary>
        private static string DetectSceneToBuild()
        {
            // OPTION 1: Use currently open scene if it's saved
            var activeScene = UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene();
            if (!string.IsNullOrEmpty(activeScene.path) && File.Exists(activeScene.path))
            {
                Debug.Log($"Using active scene: {activeScene.path}");
                return activeScene.path;
            }

            // OPTION 2: Look for preferred scene name
            string preferredPath = Path.Combine(SCENES_FOLDER, PREFERRED_SCENE_NAME);
            if (File.Exists(preferredPath))
            {
                Debug.Log($"Using preferred scene: {preferredPath}");
                return preferredPath;
            }

            // OPTION 3: Find any .unity file in Scenes folder
            if (Directory.Exists(SCENES_FOLDER))
            {
                string[] sceneFiles = Directory.GetFiles(SCENES_FOLDER, "*.unity", SearchOption.TopDirectoryOnly);
                if (sceneFiles.Length > 0)
                {
                    // Sort by last write time to get most recent
                    string mostRecent = sceneFiles.OrderByDescending(f => File.GetLastWriteTime(f)).First();
                    Debug.Log($"Using most recent scene: {mostRecent}");
                    return mostRecent;
                }
            }

            // OPTION 4: Check build settings
            string[] scenesInBuildSettings = EditorBuildSettings.scenes
                .Where(s => s.enabled && !string.IsNullOrEmpty(s.path))
                .Select(s => s.path)
                .ToArray();

            if (scenesInBuildSettings.Length > 0)
            {
                Debug.Log($"Using first scene from build settings: {scenesInBuildSettings[0]}");
                return scenesInBuildSettings[0];
            }

            // FAILURE: No scene found
            return null;
        }

        /// <summary>
        /// Configures optimal WebGL player settings
        /// </summary>
        private static void ConfigureWebGLSettings(bool isDevelopment)
        {
            // Compression
            PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Brotli;
            PlayerSettings.WebGL.decompressionFallback = true;

            // Memory
            PlayerSettings.WebGL.memorySize = 2048; // 2GB for Kelly's high-quality assets

            // Code optimization
            PlayerSettings.WebGL.exceptionSupport = isDevelopment ? 
                WebGLExceptionSupport.FullWithStacktrace : 
                WebGLExceptionSupport.None;

            // Data caching
            PlayerSettings.WebGL.dataCaching = true;

            // Stripping (production only)
            if (!isDevelopment)
            {
                PlayerSettings.stripEngineCode = true;
                // Note: managedStrippingLevel is set via Player Settings UI in Unity 6
            }

            // Graphics
            PlayerSettings.SetGraphicsAPIs(BuildTarget.WebGL, new[] { UnityEngine.Rendering.GraphicsDeviceType.OpenGLES3 });
            
            Debug.Log("✓ WebGL settings configured");
        }

        /// <summary>
        /// Command-line build support
        /// Usage: Unity -quit -batchmode -executeMethod KellySetup.BuildWebGL.CommandLineBuild [-buildDevelopment]
        /// </summary>
        public static void CommandLineBuild()
        {
            bool isDevelopment = Environment.GetCommandLineArgs().Contains("-buildDevelopment");
            Debug.Log($"Command-line build started (Development: {isDevelopment})");
            BuildWebGLInternal(isDevelopment);
        }

        private static string FormatBytes(ulong bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB" };
            double len = bytes;
            int order = 0;
            while (len >= 1024 && order < sizes.Length - 1)
            {
                order++;
                len = len / 1024;
            }
            return $"{len:0.##} {sizes[order]}";
        }

        /// <summary>
        /// Diagnostic tool to show what scene would be used
        /// </summary>
        [MenuItem("Kelly/Build/🔍 Show Scene Detection", false, 902)]
        public static void ShowSceneDetection()
        {
            Debug.Log("╔══════════════════════════════════════════════════════╗");
            Debug.Log("║             SCENE DETECTION DIAGNOSTIC               ║");
            Debug.Log("╚══════════════════════════════════════════════════════╝");
            Debug.Log("");

            string detectedScene = DetectSceneToBuild();
            
            if (string.IsNullOrEmpty(detectedScene))
            {
                Debug.LogError("✗ NO VALID SCENE FOUND!");
                Debug.LogError("");
                Debug.LogError("TROUBLESHOOTING:");
                Debug.LogError("1. Check if Assets/Scenes/ folder exists");
                Debug.LogError("2. Ensure at least one .unity file is present");
                Debug.LogError("3. Create a new scene and save it to Assets/Scenes/KellyMain.unity");
            }
            else
            {
                Debug.Log($"✓ Scene to be built: {detectedScene}");
                Debug.Log($"  File size: {new FileInfo(detectedScene).Length / 1024} KB");
                Debug.Log($"  Last modified: {File.GetLastWriteTime(detectedScene)}");
                Debug.Log("");
                Debug.Log("This scene will be used for the next build.");
            }

            Debug.Log("");
        }
    }
}
#endif
