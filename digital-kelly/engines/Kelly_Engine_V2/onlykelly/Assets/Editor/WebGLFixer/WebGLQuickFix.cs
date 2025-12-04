using UnityEngine;
using UnityEditor;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using System.IO;
using System.Collections.Generic;

/// <summary>
/// ONE-CLICK WebGL FIX for Curious Kelly
/// 
/// This script fixes the gray/clay appearance by:
/// 1. Switching to Mobile URP (Forward Rendering - WebGL compatible)
/// 2. Converting Reallusion shaders to URP/Lit
/// 3. Building optimized WebGL
/// 
/// Usage: Window > Kelly WebGL > Quick Fix
/// </summary>
public class WebGLQuickFix : EditorWindow
{
    private static readonly string MOBILE_RP_GUID = "5e6cbd92db86f4b18aec3ed561671858"; // Mobile_RPAsset
    private static readonly string BUILD_PATH = "Builds/Kelly_Web_Build";
    
    private Vector2 scrollPos;
    private bool step1Done = false;
    private bool step2Done = false;
    private bool step3Done = false;
    private int materialsFixed = 0;
    
    [MenuItem("Window/Kelly WebGL/Quick Fix (One-Click)")]
    public static void ShowWindow()
    {
        var window = GetWindow<WebGLQuickFix>("Kelly WebGL Fix");
        window.minSize = new Vector2(500, 600);
        window.Show();
    }
    
    [MenuItem("Window/Kelly WebGL/1. Fix Graphics Settings")]
    public static void FixGraphicsSettings()
    {
        // Load Mobile URP Asset
        string assetPath = AssetDatabase.GUIDToAssetPath(MOBILE_RP_GUID);
        if (string.IsNullOrEmpty(assetPath))
        {
            assetPath = "Assets/Settings/Mobile_RPAsset.asset";
        }
        
        var mobileRP = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(assetPath);
        
        if (mobileRP == null)
        {
            Debug.LogError("[WebGL Fix] Mobile_RPAsset not found! Looking for it...");
            string[] guids = AssetDatabase.FindAssets("Mobile_RPAsset t:UniversalRenderPipelineAsset");
            if (guids.Length > 0)
            {
                assetPath = AssetDatabase.GUIDToAssetPath(guids[0]);
                mobileRP = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(assetPath);
            }
        }
        
        if (mobileRP != null)
        {
            // Set as default render pipeline
            GraphicsSettings.defaultRenderPipeline = mobileRP;
            QualitySettings.renderPipeline = mobileRP;
            
            Debug.Log($"✅ [WebGL Fix] Graphics Settings fixed! Now using: {mobileRP.name}");
            Debug.Log("   - Render Mode: Forward (WebGL Compatible)");
            EditorUtility.DisplayDialog("Success!", 
                $"Graphics settings updated to use {mobileRP.name}\n\nThis enables Forward Rendering which is WebGL compatible.", 
                "OK");
        }
        else
        {
            Debug.LogError("[WebGL Fix] Could not find Mobile_RPAsset!");
            EditorUtility.DisplayDialog("Error", 
                "Could not find Mobile_RPAsset.asset\n\nPlease ensure it exists in Assets/Settings/", 
                "OK");
        }
    }
    
    [MenuItem("Window/Kelly WebGL/2. Fix Materials (Reallusion → URP Lit)")]
    public static int FixMaterials()
    {
        int fixedCount = 0;
        string[] materialGuids = AssetDatabase.FindAssets("t:Material", new[] { "Assets" });
        
        // Get URP Lit shader
        Shader urpLit = Shader.Find("Universal Render Pipeline/Lit");
        Shader urpSimpleLit = Shader.Find("Universal Render Pipeline/Simple Lit");
        
        if (urpLit == null)
        {
            Debug.LogError("[WebGL Fix] URP Lit shader not found!");
            return 0;
        }
        
        List<string> fixedMaterials = new List<string>();
        
        foreach (string guid in materialGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            Material mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            
            if (mat == null || mat.shader == null) continue;
            
            string shaderName = mat.shader.name;
            
            // Skip already compatible shaders
            if (shaderName.StartsWith("Universal Render Pipeline/") ||
                shaderName.StartsWith("Hidden/") ||
                shaderName == "Standard")
            {
                continue;
            }
            
            // Fix Reallusion and incompatible shaders
            if (shaderName.Contains("Reallusion") || 
                shaderName.Contains("RL_") ||
                shaderName.Contains("Shader Graph") ||
                shaderName.Contains("CCiC"))
            {
                // Preserve textures before changing shader
                Texture albedo = mat.HasProperty("_BaseMap") ? mat.GetTexture("_BaseMap") : 
                                 mat.HasProperty("_MainTex") ? mat.GetTexture("_MainTex") :
                                 mat.HasProperty("_DiffuseMap") ? mat.GetTexture("_DiffuseMap") : null;
                                 
                Texture normal = mat.HasProperty("_BumpMap") ? mat.GetTexture("_BumpMap") :
                                 mat.HasProperty("_NormalMap") ? mat.GetTexture("_NormalMap") : null;
                
                Color baseColor = mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor") :
                                  mat.HasProperty("_Color") ? mat.GetColor("_Color") : Color.white;
                
                float smoothness = mat.HasProperty("_Smoothness") ? mat.GetFloat("_Smoothness") : 0.5f;
                float metallic = mat.HasProperty("_Metallic") ? mat.GetFloat("_Metallic") : 0f;
                
                // Determine best shader based on material name
                Shader targetShader = urpLit;
                
                // Use Simple Lit for hair (faster, looks good)
                if (path.ToLower().Contains("hair") || path.ToLower().Contains("scalp"))
                {
                    targetShader = urpSimpleLit ?? urpLit;
                }
                
                // Apply new shader
                mat.shader = targetShader;
                
                // Restore textures
                if (albedo != null)
                {
                    if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", albedo);
                    if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", albedo);
                }
                
                if (normal != null && mat.HasProperty("_BumpMap"))
                {
                    mat.SetTexture("_BumpMap", normal);
                }
                
                if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", baseColor);
                if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", smoothness);
                if (mat.HasProperty("_Metallic")) mat.SetFloat("_Metallic", metallic);
                
                EditorUtility.SetDirty(mat);
                fixedMaterials.Add(mat.name);
                fixedCount++;
                
                Debug.Log($"✅ Fixed material: {mat.name} ({shaderName} → {targetShader.name})");
            }
        }
        
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        
        if (fixedCount > 0)
        {
            Debug.Log($"✅ [WebGL Fix] Fixed {fixedCount} materials!");
            EditorUtility.DisplayDialog("Materials Fixed!", 
                $"Converted {fixedCount} materials to URP/Lit:\n\n" +
                string.Join("\n", fixedMaterials.GetRange(0, Mathf.Min(10, fixedMaterials.Count))) +
                (fixedMaterials.Count > 10 ? $"\n... and {fixedMaterials.Count - 10} more" : ""),
                "OK");
        }
        else
        {
            Debug.Log("[WebGL Fix] No materials needed fixing.");
            EditorUtility.DisplayDialog("Materials Check", 
                "No Reallusion materials found that need conversion.\n\nMaterials may already be fixed or using compatible shaders.", 
                "OK");
        }
        
        return fixedCount;
    }
    
    [MenuItem("Window/Kelly WebGL/3. Build WebGL")]
    public static void BuildWebGL()
    {
        // Ensure WebGL is the target
        if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.WebGL)
        {
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.WebGL, BuildTarget.WebGL);
        }
        
        // Get scenes
        string[] scenes = new string[] { "Assets/Scenes/KellyMain.unity" };
        
        // Check if scene exists
        if (!File.Exists(Path.Combine(Application.dataPath, "../Assets/Scenes/KellyMain.unity")))
        {
            // Try alternative
            scenes = new string[] { "Assets/KellyMain.unity" };
        }
        
        // Build options
        BuildPlayerOptions buildOptions = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = BUILD_PATH,
            target = BuildTarget.WebGL,
            options = BuildOptions.None // Use BuildOptions.Development for debug
        };
        
        Debug.Log("[WebGL Fix] Starting WebGL build...");
        Debug.Log($"   Output: {BUILD_PATH}");
        
        var report = BuildPipeline.BuildPlayer(buildOptions);
        
        if (report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded)
        {
            Debug.Log($"✅ [WebGL Fix] Build succeeded! Size: {report.summary.totalSize / 1024 / 1024}MB");
            Debug.Log($"   Output: {Path.GetFullPath(BUILD_PATH)}");
            
            // Open folder
            EditorUtility.RevealInFinder(BUILD_PATH);
            
            EditorUtility.DisplayDialog("Build Complete!", 
                $"WebGL build succeeded!\n\n" +
                $"Output: {Path.GetFullPath(BUILD_PATH)}\n" +
                $"Size: {report.summary.totalSize / 1024 / 1024}MB\n\n" +
                "Copy the Build folder contents to:\npublic/unity/kelly-live/Build/",
                "OK");
        }
        else
        {
            Debug.LogError($"[WebGL Fix] Build failed: {report.summary.result}");
            EditorUtility.DisplayDialog("Build Failed", 
                $"WebGL build failed!\n\nCheck Console for errors.\n\nResult: {report.summary.result}", 
                "OK");
        }
    }
    
    [MenuItem("Window/Kelly WebGL/⚡ DO EVERYTHING (Recommended)")]
    public static void DoEverything()
    {
        Debug.Log("═══════════════════════════════════════════════════════════");
        Debug.Log("   KELLY WEBGL QUICK FIX - STARTING");
        Debug.Log("═══════════════════════════════════════════════════════════");
        
        // Step 1: Fix Graphics
        Debug.Log("\n[Step 1/3] Fixing Graphics Settings...");
        FixGraphicsSettings();
        
        // Step 2: Fix Materials
        Debug.Log("\n[Step 2/3] Fixing Materials...");
        int materialsConverted = FixMaterials();
        
        // Step 3: Build
        Debug.Log("\n[Step 3/3] Building WebGL...");
        
        bool doBuild = EditorUtility.DisplayDialog("Ready to Build?",
            $"Graphics settings: ✅ Fixed\n" +
            $"Materials: ✅ {materialsConverted} converted\n\n" +
            "Ready to build WebGL?\n\n" +
            "(This may take a few minutes)",
            "Build Now", "Skip Build");
        
        if (doBuild)
        {
            BuildWebGL();
        }
        
        Debug.Log("\n═══════════════════════════════════════════════════════════");
        Debug.Log("   KELLY WEBGL QUICK FIX - COMPLETE");
        Debug.Log("═══════════════════════════════════════════════════════════");
    }
    
    void OnGUI()
    {
        scrollPos = EditorGUILayout.BeginScrollView(scrollPos);
        
        // Header
        GUILayout.Space(10);
        GUIStyle headerStyle = new GUIStyle(EditorStyles.boldLabel);
        headerStyle.fontSize = 18;
        headerStyle.alignment = TextAnchor.MiddleCenter;
        GUILayout.Label("🎭 Kelly WebGL Quick Fix", headerStyle);
        
        GUILayout.Space(5);
        GUIStyle subStyle = new GUIStyle(EditorStyles.label);
        subStyle.alignment = TextAnchor.MiddleCenter;
        subStyle.wordWrap = true;
        GUILayout.Label("Fix the gray/clay avatar in 3 clicks", subStyle);
        
        GUILayout.Space(20);
        
        // One-Click Button
        GUI.backgroundColor = new Color(0.2f, 0.8f, 0.2f);
        if (GUILayout.Button("⚡ DO EVERYTHING (Recommended)", GUILayout.Height(50)))
        {
            DoEverything();
        }
        GUI.backgroundColor = Color.white;
        
        GUILayout.Space(20);
        EditorGUILayout.HelpBox(
            "Or run each step manually:", 
            MessageType.Info);
        
        GUILayout.Space(10);
        
        // Step 1
        EditorGUILayout.BeginHorizontal();
        GUILayout.Label(step1Done ? "✅" : "1️⃣", GUILayout.Width(25));
        if (GUILayout.Button("Fix Graphics Settings", GUILayout.Height(35)))
        {
            FixGraphicsSettings();
            step1Done = true;
        }
        EditorGUILayout.EndHorizontal();
        EditorGUILayout.HelpBox("Switches to Mobile URP (Forward Rendering)", MessageType.None);
        
        GUILayout.Space(10);
        
        // Step 2
        EditorGUILayout.BeginHorizontal();
        GUILayout.Label(step2Done ? "✅" : "2️⃣", GUILayout.Width(25));
        if (GUILayout.Button("Fix Materials", GUILayout.Height(35)))
        {
            materialsFixed = FixMaterials();
            step2Done = true;
        }
        EditorGUILayout.EndHorizontal();
        EditorGUILayout.HelpBox("Converts Reallusion shaders to URP/Lit", MessageType.None);
        if (step2Done)
        {
            GUILayout.Label($"   → {materialsFixed} materials converted");
        }
        
        GUILayout.Space(10);
        
        // Step 3
        EditorGUILayout.BeginHorizontal();
        GUILayout.Label(step3Done ? "✅" : "3️⃣", GUILayout.Width(25));
        if (GUILayout.Button("Build WebGL", GUILayout.Height(35)))
        {
            BuildWebGL();
            step3Done = true;
        }
        EditorGUILayout.EndHorizontal();
        EditorGUILayout.HelpBox($"Builds to: {BUILD_PATH}", MessageType.None);
        
        GUILayout.Space(20);
        
        // After Build Instructions
        EditorGUILayout.HelpBox(
            "After building:\n\n" +
            "1. Copy build files to: public/unity/kelly-live/Build/\n" +
            "2. Test locally: http://localhost:3000/unity-test.html\n" +
            "3. Kelly should now have proper colors!", 
            MessageType.Info);
        
        GUILayout.Space(10);
        
        // Current Status
        EditorGUILayout.LabelField("Current Render Pipeline:", EditorStyles.boldLabel);
        var currentRP = GraphicsSettings.defaultRenderPipeline;
        if (currentRP != null)
        {
            EditorGUILayout.LabelField($"   {currentRP.name}");
            
            // Check if it's Forward or Deferred
            if (currentRP.name.Contains("Mobile"))
            {
                EditorGUILayout.HelpBox("✅ Using Mobile URP (Forward) - WebGL Compatible!", MessageType.Info);
            }
            else if (currentRP.name.Contains("PC"))
            {
                EditorGUILayout.HelpBox("⚠️ Using PC URP (may use Deferred) - May not work on WebGL!", MessageType.Warning);
            }
        }
        else
        {
            EditorGUILayout.LabelField("   None (Built-in)");
        }
        
        EditorGUILayout.EndScrollView();
    }
}

