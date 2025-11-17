using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Kelly.Editor
{
    /// <summary>
    /// Deterministic WebGL builder for the Kelly avatar demo.
    /// Usage (CLI):
    /// Unity.exe -quit -batchmode -projectPath <path> -executeMethod Kelly.Editor.WebGLBuild.Build
    /// </summary>
    public static class WebGLBuild
    {
        private const string DefaultBuildLabel = "kelly-v1";

        [MenuItem("Kelly/Build/WebGL (iframe bundle)")]
        public static void Build()
        {
            var scenes = EditorBuildSettings.scenes
                .Where(scene => scene.enabled)
                .Select(scene => scene.path)
                .ToArray();

            if (scenes.Length == 0)
            {
                Debug.LogError("[Kelly WebGL] No enabled scenes found. Please add Main.unity to Build Settings.");
                return;
            }

            var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            var buildLabel = Environment.GetEnvironmentVariable("KELLY_WEBGL_VERSION");
            if (string.IsNullOrWhiteSpace(buildLabel))
            {
                buildLabel = DefaultBuildLabel;
            }

            var outputDirectory = Path.Combine(projectRoot, "Builds", "WebGL", buildLabel);
            Directory.CreateDirectory(outputDirectory);

            ConfigurePlayerSettings();

            Debug.Log($"[Kelly WebGL] Building scenes: {string.Join(", ", scenes)}");
            Debug.Log($"[Kelly WebGL] Output path: {outputDirectory}");

            var buildPlayerOptions = new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = outputDirectory,
                target = BuildTarget.WebGL,
                options = BuildOptions.None
            };

            BuildReport report = BuildPipeline.BuildPlayer(buildPlayerOptions);
            BuildSummary summary = report.summary;

            if (summary.result == BuildResult.Succeeded)
            {
                var dataSizeMb = (summary.totalSize / 1024f / 1024f).ToString("F2");
                Debug.Log($"✅ [Kelly WebGL] Build succeeded. Size: {dataSizeMb} MB, Time: {summary.totalTime.TotalSeconds:F1}s");
            }
            else
            {
                Debug.LogError($"❌ [Kelly WebGL] Build failed: {summary.result}");
                foreach (var step in report.steps)
                {
                    foreach (var message in step.messages)
                    {
                        if (message.type == LogType.Error || message.type == LogType.Exception)
                        {
                            Debug.LogError($"[Kelly WebGL] {message.content}");
                        }
                    }
                }
            }
        }

        private static void ConfigurePlayerSettings()
        {
            PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Gzip;
            PlayerSettings.WebGL.decompressionFallback = true;
            PlayerSettings.WebGL.dataCaching = true;
            PlayerSettings.WebGL.nameFilesAsHashes = true;
            PlayerSettings.WebGL.exceptionSupport = WebGLExceptionSupport.FullWithStackTrace;
            PlayerSettings.WebGL.linkerTarget = WebGLLinkerTarget.Wasm;
            PlayerSettings.defaultWebScreenWidth = 1920;
            PlayerSettings.defaultWebScreenHeight = 1080;
            PlayerSettings.defaultScreenWidth = 1920;
            PlayerSettings.defaultScreenHeight = 1080;
            PlayerSettings.SplashScreen.showUnityLogo = false;
            PlayerSettings.stripEngineCode = true;
            EditorUserBuildSettings.development = false;
            EditorUserBuildSettings.webGLLinkerTarget = WebGLLinkerTarget.Wasm;
        }
    }
}

