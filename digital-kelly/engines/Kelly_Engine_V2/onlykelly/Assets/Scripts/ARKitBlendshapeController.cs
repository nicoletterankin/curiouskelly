using UnityEngine;
using System.Collections.Generic;

public class ARKitBlendshapeController : MonoBehaviour
{
    [Header("References")]
    public SkinnedMeshRenderer headRenderer;
    
    [Header("Blendshape Mapping")]
    public Dictionary<string, int> blendshapeMap = new Dictionary<string, int>();
    
    [Header("Runtime Values")]
    [Range(0f, 1f)] public float testValue = 0f;
    public string testBlendshape = "V_Open";
    
    void Start()
    {
        InitializeBlendshapeMap();
    }
    
    public void InitializeBlendshapeMap()
    {
        if (headRenderer == null)
        {
            headRenderer = GetComponentInChildren<SkinnedMeshRenderer>();
            if (headRenderer == null)
            {
                Debug.LogError("[ARKitBlendshapeController] No SkinnedMeshRenderer found!");
                return;
            }
        }
        
        Mesh mesh = headRenderer.sharedMesh;
        int count = mesh.blendShapeCount;
        
        blendshapeMap.Clear();
        
        for (int i = 0; i < count; i++)
        {
            string name = mesh.GetBlendShapeName(i);
            blendshapeMap[name] = i;
        }
        
        Debug.Log($"[ARKitBlendshapeController] Initialized {blendshapeMap.Count} blendshapes");
    }
    
    private void Update()
    {
        if (Application.isEditor && testValue > 0f)
        {
            SetBlendshape(testBlendshape, testValue * 100f);
        }
    }
    
    public void SetBlendshape(string name, float value)
    {
        if (headRenderer == null) return;
        
        // Direct match
        if (blendshapeMap.TryGetValue(name, out int index))
        {
            headRenderer.SetBlendShapeWeight(index, value);
            return;
        }
        
        // Partial match fallback
        foreach (var kvp in blendshapeMap)
        {
            if (kvp.Key.ToLower().Contains(name.ToLower()) || 
                name.ToLower().Contains(kvp.Key.ToLower()))
            {
                headRenderer.SetBlendShapeWeight(kvp.Value, value);
                return;
            }
        }
        
        Debug.LogWarning($"[ARKitBlendshapeController] Blendshape '{name}' not found");
    }
    
    public void SetAllBlendshapes(Dictionary<string, float> values)
    {
        foreach (var kvp in values)
        {
            SetBlendshape(kvp.Key, kvp.Value * 100f);
        }
    }
    
    public void ResetAll()
    {
        if (headRenderer == null) return;
        
        for (int i = 0; i < headRenderer.sharedMesh.blendShapeCount; i++)
        {
            headRenderer.SetBlendShapeWeight(i, 0f);
        }
    }
}

