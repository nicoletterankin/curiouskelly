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
    public string testBlendshape = "mouthSmile_L";
    
    private void Update()
    {
        // Test in editor
        if (Application.isEditor && testValue > 0f)
        {
            SetBlendshape(testBlendshape, testValue * 100f);
        }
    }
    
    public void SetBlendshape(string arkitName, float value)
    {
        if (blendshapeMap.ContainsKey(arkitName))
        {
            int index = blendshapeMap[arkitName];
            headRenderer.SetBlendShapeWeight(index, value);
        }
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
        for (int i = 0; i < headRenderer.sharedMesh.blendShapeCount; i++)
        {
            headRenderer.SetBlendShapeWeight(i, 0f);
        }
    }
}

