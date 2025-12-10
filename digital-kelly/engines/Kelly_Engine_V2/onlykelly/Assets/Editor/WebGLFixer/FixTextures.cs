using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;

/// <summary>
/// Fix missing textures on Kelly's materials
/// Finds diffuse textures and assigns them to URP/Lit materials
/// </summary>
public class FixTextures : EditorWindow
{
    [MenuItem("Window/Kelly WebGL/🔧 Fix Missing Textures")]
    public static void FixAllTextures()
    {
        int fixedCount = 0;
        
        // Find all materials
        string[] materialGuids = AssetDatabase.FindAssets("t:Material", new[] { "Assets" });
        
        // Build a lookup of diffuse textures
        Dictionary<string, Texture2D> diffuseTextures = new Dictionary<string, Texture2D>();
        string[] textureGuids = AssetDatabase.FindAssets("Diffuse t:Texture2D", new[] { "Assets" });
        
        foreach (string guid in textureGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            if (tex != null)
            {
                // Extract base name (e.g., "Std_Skin_Head" from "Std_Skin_Head_Diffuse.png")
                string fileName = Path.GetFileNameWithoutExtension(path);
                string baseName = fileName.Replace("_Diffuse", "").Replace(" 1", "").Trim();
                
                if (!diffuseTextures.ContainsKey(baseName))
                {
                    diffuseTextures[baseName] = tex;
                    Debug.Log($"[FixTextures] Found texture: {baseName} -> {path}");
                }
            }
        }
        
        Debug.Log($"[FixTextures] Found {diffuseTextures.Count} diffuse textures");
        
        // Fix each material
        foreach (string guid in materialGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            Material mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            
            if (mat == null) continue;
            
            // Skip if not URP/Lit
            if (mat.shader == null || !mat.shader.name.Contains("Universal Render Pipeline/Lit"))
                continue;
            
            // Check if BaseMap is missing
            Texture currentTex = mat.HasProperty("_BaseMap") ? mat.GetTexture("_BaseMap") : null;
            if (currentTex != null) continue; // Already has texture
            
            // Try to find matching diffuse texture
            string matName = mat.name.Replace("_URP", "").Replace("Template_", "").Trim();
            
            // Try exact match first
            if (diffuseTextures.TryGetValue(matName, out Texture2D diffuseTex))
            {
                mat.SetTexture("_BaseMap", diffuseTex);
                mat.SetColor("_BaseColor", Color.white);
                EditorUtility.SetDirty(mat);
                fixedCount++;
                Debug.Log($"✅ Fixed: {mat.name} <- {diffuseTex.name}");
            }
            else
            {
                // Try partial match
                foreach (var kvp in diffuseTextures)
                {
                    if (matName.Contains(kvp.Key) || kvp.Key.Contains(matName))
                    {
                        mat.SetTexture("_BaseMap", kvp.Value);
                        mat.SetColor("_BaseColor", Color.white);
                        EditorUtility.SetDirty(mat);
                        fixedCount++;
                        Debug.Log($"✅ Fixed (partial): {mat.name} <- {kvp.Value.name}");
                        break;
                    }
                }
            }
        }
        
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        
        Debug.Log($"[FixTextures] Fixed {fixedCount} materials with missing textures");
        
        EditorUtility.DisplayDialog("Textures Fixed", 
            $"Assigned diffuse textures to {fixedCount} materials.\n\n" +
            "Now rebuild: Window > Kelly WebGL > 3. Build WebGL",
            "OK");
    }
    
    [MenuItem("Window/Kelly WebGL/🎨 Auto-Fix Kelly Materials (Complete)")]
    public static void CompleteKellyFix()
    {
        Debug.Log("═══════════════════════════════════════════════════════════");
        Debug.Log("   COMPLETE KELLY MATERIAL FIX");
        Debug.Log("═══════════════════════════════════════════════════════════");
        
        int totalFixed = 0;
        
        // Get URP Lit shader
        Shader urpLit = Shader.Find("Universal Render Pipeline/Lit");
        if (urpLit == null)
        {
            Debug.LogError("URP Lit shader not found!");
            return;
        }
        
        // Build texture lookup
        Dictionary<string, Texture2D> textureLookup = new Dictionary<string, Texture2D>();
        string[] textureGuids = AssetDatabase.FindAssets("t:Texture2D", new[] { "Assets" });
        
        foreach (string guid in textureGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (path.Contains("Reallusion/CCiC Unity Tools")) continue; // Skip template textures
            
            Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            if (tex != null)
            {
                string fileName = Path.GetFileNameWithoutExtension(path);
                
                // Categorize by type
                if (fileName.Contains("Diffuse") || fileName.Contains("_d"))
                {
                    string baseName = fileName.Replace("_Diffuse", "").Replace(" 1", "").Trim();
                    textureLookup[$"{baseName}_Diffuse"] = tex;
                }
                else if (fileName.Contains("Normal") || fileName.Contains("_n"))
                {
                    string baseName = fileName.Replace("_Normal", "").Replace(" 1", "").Trim();
                    textureLookup[$"{baseName}_Normal"] = tex;
                }
            }
        }
        
        Debug.Log($"Found {textureLookup.Count} textures");
        
        // Get all Kelly-related materials
        string[] matPaths = new string[]
        {
            "Assets/Std_Skin_Head.mat",
            "Assets/Std_Skin_Body.mat",
            "Assets/Std_Skin_Arm.mat",
            "Assets/Std_Skin_Leg.mat",
            "Assets/Std_Eye_L.mat",
            "Assets/Std_Eye_R.mat",
            "Assets/Std_Cornea_L.mat",
            "Assets/Std_Cornea_R.mat",
            "Assets/Std_Eyelash.mat",
            "Assets/Std_Tongue.mat",
            "Assets/Std_Upper_Teeth.mat",
            "Assets/Std_Lower_Teeth.mat",
            "Assets/Std_Nails.mat",
            "Assets/Hair_L_Transparency.mat",
            "Assets/Hair_R_Transparency.mat",
            "Assets/Scalp_Transparency.mat",
            "Assets/Layered_sweater.mat",
            "Assets/Pants.mat",
            "Assets/Canvas_shoes.mat",
            "Assets/Jeans.mat",
        };
        
        foreach (string matPath in matPaths)
        {
            Material mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
            if (mat == null) 
            {
                // Try without Assets/ prefix
                mat = AssetDatabase.LoadAssetAtPath<Material>(matPath.Replace("Assets/", ""));
                if (mat == null) continue;
            }
            
            string baseName = mat.name;
            
            // Set shader to URP Lit
            mat.shader = urpLit;
            
            // Find and assign diffuse texture
            string diffuseKey = $"{baseName}_Diffuse";
            if (textureLookup.TryGetValue(diffuseKey, out Texture2D diffuseTex))
            {
                mat.SetTexture("_BaseMap", diffuseTex);
                mat.SetColor("_BaseColor", Color.white);
                Debug.Log($"✅ {baseName}: Diffuse assigned");
                totalFixed++;
            }
            else
            {
                // Try finding by partial match
                foreach (var kvp in textureLookup)
                {
                    if (kvp.Key.Contains(baseName) && kvp.Key.Contains("Diffuse"))
                    {
                        mat.SetTexture("_BaseMap", kvp.Value);
                        mat.SetColor("_BaseColor", Color.white);
                        Debug.Log($"✅ {baseName}: Diffuse assigned (partial match)");
                        totalFixed++;
                        break;
                    }
                }
            }
            
            // Find and assign normal texture
            string normalKey = $"{baseName}_Normal";
            if (textureLookup.TryGetValue(normalKey, out Texture2D normalTex))
            {
                mat.SetTexture("_BumpMap", normalTex);
                mat.SetFloat("_BumpScale", 1.0f);
            }
            
            // Set reasonable defaults for skin
            if (baseName.Contains("Skin"))
            {
                mat.SetFloat("_Smoothness", 0.3f);
                mat.SetFloat("_Metallic", 0f);
            }
            
            EditorUtility.SetDirty(mat);
        }
        
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        
        Debug.Log($"\n✅ Fixed {totalFixed} materials!");
        Debug.Log("═══════════════════════════════════════════════════════════");
        
        EditorUtility.DisplayDialog("Kelly Materials Fixed!", 
            $"Fixed {totalFixed} materials with proper textures.\n\n" +
            "Now rebuild:\nWindow > Kelly WebGL > 3. Build WebGL",
            "OK");
    }
}




