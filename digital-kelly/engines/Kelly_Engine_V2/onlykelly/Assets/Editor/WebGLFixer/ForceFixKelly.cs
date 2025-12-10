using UnityEngine;
using UnityEditor;
using System.IO;

public class ForceFixKelly : MonoBehaviour
{
    [MenuItem("Kelly/FORCE FIX ALL MATERIALS NOW")]
    public static void FixNow()
    {
        Shader litShader = Shader.Find("Universal Render Pipeline/Lit");
        if (litShader == null)
        {
            Debug.LogError("URP Lit shader not found!");
            return;
        }

        int count = 0;
        
        // Find all materials in project
        string[] guids = AssetDatabase.FindAssets("t:Material");
        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            Material mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            
            if (mat == null) continue;
            if (mat.shader == null) continue;
            
            // Skip if already simple URP/Lit
            if (mat.shader.name == "Universal Render Pipeline/Lit") continue;
            if (mat.shader.name == "Universal Render Pipeline/Simple Lit") continue;
            
            // Skip Unity built-in
            if (path.StartsWith("Packages/")) continue;
            if (mat.name.StartsWith("Default-")) continue;
            
            string matName = mat.name;
            
            // Find matching diffuse texture
            Texture2D diffuse = FindTexture(matName, "Diffuse");
            if (diffuse == null) diffuse = FindTexture(matName, "BDiffuseBlur");
            if (diffuse == null) diffuse = FindTexture(matName, "_d");
            
            // Change shader
            mat.shader = litShader;
            
            // Assign texture
            if (diffuse != null)
            {
                mat.SetTexture("_BaseMap", diffuse);
                mat.SetColor("_BaseColor", Color.white);
                Debug.Log($"Fixed: {matName} with {diffuse.name}");
            }
            else
            {
                mat.SetColor("_BaseColor", new Color(0.8f, 0.7f, 0.6f)); // Skin-ish fallback
                Debug.Log($"Fixed: {matName} (no texture found)");
            }
            
            mat.SetFloat("_Smoothness", 0.3f);
            mat.SetFloat("_Metallic", 0f);
            
            EditorUtility.SetDirty(mat);
            count++;
        }
        
        AssetDatabase.SaveAssets();
        Debug.Log($"DONE! Fixed {count} materials. Now build WebGL.");
        
        EditorUtility.DisplayDialog("FIXED", 
            $"Forced {count} materials to URP/Lit.\n\nNow: Window > Kelly WebGL > 3. Build WebGL", "OK");
    }
    
    static Texture2D FindTexture(string matName, string suffix)
    {
        string[] patterns = new string[]
        {
            $"{matName}_{suffix}",
            $"{matName}{suffix}",
            $"{matName.Replace("Std_", "")}_{suffix}",
        };
        
        foreach (string pattern in patterns)
        {
            string[] guids = AssetDatabase.FindAssets($"{pattern} t:Texture2D");
            if (guids.Length > 0)
            {
                string path = AssetDatabase.GUIDToAssetPath(guids[0]);
                return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            }
        }
        return null;
    }
}




