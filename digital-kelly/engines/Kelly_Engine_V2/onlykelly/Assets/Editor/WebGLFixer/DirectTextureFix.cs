using UnityEngine;
using UnityEditor;

public class DirectTextureFix : MonoBehaviour
{
    [MenuItem("Kelly/DIRECT TEXTURE FIX")]
    public static void Fix()
    {
        Shader lit = Shader.Find("Universal Render Pipeline/Lit");
        int count = 0;
        
        // Direct mappings: material name contains -> texture name contains
        string[,] mappings = new string[,] {
            {"Skin_Head", "Std_Skin_Head"},
            {"Skin_Body", "Std_Skin_Body"},
            {"Skin_Arm", "Std_Skin_Arm"},
            {"Skin_Leg", "Std_Skin_Leg"},
            {"Eye_L", "Std_Eye_L"},
            {"Eye_R", "Std_Eye_R"},
            {"Cornea", "Std_Cornea"},
            {"Eyelash", "Std_Eyelash"},
            {"Tongue", "Std_Tongue"},
            {"Teeth", "Std_Teeth"},
            {"Upper_Teeth", "Std_Upper_Teeth"},
            {"Lower_Teeth", "Std_Lower_Teeth"},
            {"Nails", "Std_Nails"},
            {"Hair", "Hair"},
            {"Scalp", "Scalp"},
            {"sweater", "sweater"},
            {"Pants", "Pants"},
            {"Jeans", "Jeans"},
            {"shoes", "shoes"},
            {"Canvas", "Canvas"},
        };
        
        string[] matGuids = AssetDatabase.FindAssets("t:Material", new[]{"Assets"});
        
        foreach (string guid in matGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            Material mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null) continue;
            
            // Change shader
            mat.shader = lit;
            
            // Find matching texture
            for (int i = 0; i < mappings.GetLength(0); i++)
            {
                if (mat.name.ToLower().Contains(mappings[i,0].ToLower()))
                {
                    Texture2D tex = FindTexture(mappings[i,1]);
                    if (tex != null)
                    {
                        mat.SetTexture("_BaseMap", tex);
                        mat.SetColor("_BaseColor", Color.white);
                        Debug.Log($"✓ {mat.name} <- {tex.name}");
                        count++;
                        break;
                    }
                }
            }
            
            EditorUtility.SetDirty(mat);
        }
        
        // Also fix materials directly on the prefab
        FixPrefabMaterials(lit, mappings, ref count);
        
        AssetDatabase.SaveAssets();
        Debug.Log($"DONE! Fixed {count} texture assignments.");
        EditorUtility.DisplayDialog("Done", $"Fixed {count} textures. Build now.", "OK");
    }
    
    static void FixPrefabMaterials(Shader lit, string[,] mappings, ref int count)
    {
        // Find Kelly prefab
        string[] prefabGuids = AssetDatabase.FindAssets("Kelly_CC5_WebGL t:Prefab");
        foreach (string guid in prefabGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (prefab == null) continue;
            
            var renderers = prefab.GetComponentsInChildren<Renderer>(true);
            foreach (var rend in renderers)
            {
                foreach (var mat in rend.sharedMaterials)
                {
                    if (mat == null) continue;
                    mat.shader = lit;
                    
                    for (int i = 0; i < mappings.GetLength(0); i++)
                    {
                        if (mat.name.ToLower().Contains(mappings[i,0].ToLower()) ||
                            rend.name.ToLower().Contains(mappings[i,0].ToLower()))
                        {
                            Texture2D tex = FindTexture(mappings[i,1]);
                            if (tex != null && mat.GetTexture("_BaseMap") == null)
                            {
                                mat.SetTexture("_BaseMap", tex);
                                mat.SetColor("_BaseColor", Color.white);
                                Debug.Log($"✓ Prefab: {mat.name} <- {tex.name}");
                                count++;
                                break;
                            }
                        }
                    }
                    EditorUtility.SetDirty(mat);
                }
            }
        }
    }
    
    static Texture2D FindTexture(string nameContains)
    {
        string[] guids = AssetDatabase.FindAssets($"{nameContains} t:Texture2D", new[]{"Assets"});
        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (path.Contains("Diffuse") || path.Contains("BDiffuse"))
            {
                return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            }
        }
        // Fallback - any texture with that name
        if (guids.Length > 0)
        {
            return AssetDatabase.LoadAssetAtPath<Texture2D>(AssetDatabase.GUIDToAssetPath(guids[0]));
        }
        return null;
    }
}



