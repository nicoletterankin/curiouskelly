using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Maps OpenAI Realtime API visemes to Audio2Face blendshapes
/// Handles real-time viseme updates for lip-sync
/// </summary>
public class VisemeMapper : MonoBehaviour
{
    [Header("References")]
    public SkinnedMeshRenderer headRenderer;
    
    [Header("Settings")]
    public float smoothingSpeed = 10f;
    public float intensity = 100f;
    
    // Viseme to blendshape mapping
    private Dictionary<string, BlendshapeTarget> visemeMap;
    private Dictionary<string, int> blendshapeIndices = new Dictionary<string, int>();
    
    // Current state
    private Dictionary<string, float> currentWeights = new Dictionary<string, float>();
    private Dictionary<string, float> targetWeights = new Dictionary<string, float>();
    
    // Performance
    private bool isInitialized = false;
    private int blendShapeCount = 0;
    
    void Start()
    {
        Initialize();
    }
    
    void Initialize()
    {
        if (isInitialized) return;
        
        if (headRenderer == null)
        {
            headRenderer = GetComponentInChildren<SkinnedMeshRenderer>();
        }
        
        if (headRenderer == null || headRenderer.sharedMesh == null)
        {
            Debug.LogError("[Viseme Mapper] SkinnedMeshRenderer not found!");
            return;
        }
        
        // Index all blendshapes
        IndexBlendshapes();
        
        // Create viseme mapping
        CreateVisemeMapping();
        
        isInitialized = true;
        Debug.Log($"[Viseme Mapper] Initialized with {visemeMap.Count} visemes mapped");
    }
    
    void IndexBlendshapes()
    {
        blendShapeCount = headRenderer.sharedMesh.blendShapeCount;
        
        for (int i = 0; i < blendShapeCount; i++)
        {
            string name = headRenderer.sharedMesh.GetBlendShapeName(i);
            string normalized = NormalizeName(name);
            
            if (!blendshapeIndices.ContainsKey(normalized))
            {
                blendshapeIndices[normalized] = i;
            }
        }
        
        Debug.Log($"[Viseme Mapper] Indexed {blendshapeIndices.Count} blendshapes");
    }
    
    string NormalizeName(string name)
    {
        return name
            .Replace(" ", "")
            .Replace("_L", "_Left")
            .Replace("_R", "_Right")
            .ToLower();
    }
    
    void CreateVisemeMapping()
    {
        visemeMap = new Dictionary<string, BlendshapeTarget>();
        
        // OpenAI Realtime API visemes → Audio2Face blendshapes
        // Based on ARKit and A2F standard blendshape names
        
        // Silence
        AddMapping("sil", "jawopen", 0f);
        
        // Consonants
        AddMapping("PP", "mouthpucker", 80f);      // P, B, M sounds
        AddMapping("FF", "mouthfunnel", 70f);      // F, V sounds
        AddMapping("TH", "tongueout", 60f);        // Th sounds
        AddMapping("DD", "jawopen", 40f);          // D, T sounds
        AddMapping("kk", "jawopen", 20f);          // K, G sounds
        AddMapping("CH", "mouthshrugupper", 50f);  // Ch, J sounds
        AddMapping("SS", "mouthsmile", 40f);       // S, Z sounds
        AddMapping("nn", "jawopen", 30f);          // N sounds
        AddMapping("RR", "mouthrollupper", 40f);   // R sounds
        
        // Vowels
        AddMapping("aa", "jawopen", 70f);          // Ah sound (cat, father)
        AddMapping("E", "mouthsmile", 60f);        // Ee sound (see, bee)
        AddMapping("I", "mouthsmile", 40f);        // Ih sound (sit, bit)
        AddMapping("O", "mouthfunnel", 60f);       // Oh sound (go, phone)
        AddMapping("U", "mouthpucker", 70f);       // Oo sound (boot, blue)
        
        // Additional vowels
        AddMapping("@", "jawopen", 35f);           // Schwa (about, sofa)
        AddMapping("e", "mouthsmile", 50f);        // Eh sound (bed, said)
        AddMapping("a", "jawopen", 50f);           // Ae sound (cat, bat)
        AddMapping("o", "mouthfunnel", 40f);       // Aw sound (caught, law)
        AddMapping("u", "mouthpucker", 50f);       // Uh sound (book, put)
        
        Debug.Log($"[Viseme Mapper] Created {visemeMap.Count} viseme mappings");
    }
    
    void AddMapping(string visemeId, string blendshapeName, float weight)
    {
        visemeMap[visemeId.ToLower()] = new BlendshapeTarget
        {
            blendshapeName = blendshapeName.ToLower(),
            targetWeight = weight
        };
    }
    
    void Update()
    {
        if (!isInitialized) return;
        
        // Smooth interpolation to target weights
        foreach (var kvp in targetWeights)
        {
            string blendshape = kvp.Key;
            float target = kvp.Value;
            
            if (!currentWeights.ContainsKey(blendshape))
            {
                currentWeights[blendshape] = 0f;
            }
            
            currentWeights[blendshape] = Mathf.Lerp(
                currentWeights[blendshape],
                target,
                smoothingSpeed * Time.deltaTime
            );
        }
        
        // Apply weights to blendshapes
        ApplyBlendshapes();
    }
    
    void ApplyBlendshapes()
    {
        foreach (var kvp in currentWeights)
        {
            string blendshapeName = kvp.Key;
            float weight = kvp.Value * (intensity / 100f);
            
            if (blendshapeIndices.TryGetValue(blendshapeName, out int index))
            {
                headRenderer.SetBlendShapeWeight(index, weight);
            }
        }
    }
    
    /// <summary>
    /// Apply viseme from OpenAI Realtime API
    /// </summary>
    public void ApplyViseme(string visemeId, float weight = 1f)
    {
        if (!isInitialized) return;
        
        visemeId = visemeId.ToLower();
        
        if (visemeMap.TryGetValue(visemeId, out BlendshapeTarget target))
        {
            // Clear all current targets (one viseme at a time)
            targetWeights.Clear();
            
            // Set new target
            targetWeights[target.blendshapeName] = target.targetWeight * weight;
        }
        else
        {
            Debug.LogWarning($"[Viseme Mapper] Unknown viseme: {visemeId}");
        }
    }
    
    /// <summary>
    /// Apply multiple visemes with blending
    /// </summary>
    public void ApplyVisemes(Dictionary<string, float> visemes)
    {
        if (!isInitialized) return;
        
        // Clear current targets
        targetWeights.Clear();
        
        // Apply each viseme with its weight
        foreach (var kvp in visemes)
        {
            string visemeId = kvp.Key.ToLower();
            float weight = kvp.Value;
            
            if (visemeMap.TryGetValue(visemeId, out BlendshapeTarget target))
            {
                string blendshape = target.blendshapeName;
                float targetWeight = target.targetWeight * weight;
                
                // Blend if multiple visemes affect same blendshape
                if (targetWeights.ContainsKey(blendshape))
                {
                    targetWeights[blendshape] = Mathf.Max(targetWeights[blendshape], targetWeight);
                }
                else
                {
                    targetWeights[blendshape] = targetWeight;
                }
            }
        }
    }
    
    /// <summary>
    /// Reset all visemes to neutral (silence)
    /// </summary>
    public void ResetVisemes()
    {
        targetWeights.Clear();
        ApplyViseme("sil");
    }
    
    /// <summary>
    /// Check if a viseme is supported
    /// </summary>
    public bool IsVisemeSupported(string visemeId)
    {
        return visemeMap.ContainsKey(visemeId.ToLower());
    }
    
    /// <summary>
    /// Get list of supported visemes
    /// </summary>
    public List<string> GetSupportedVisemes()
    {
        return new List<string>(visemeMap.Keys);
    }
}

/// <summary>
/// Blendshape target for a viseme
/// </summary>
[System.Serializable]
public class BlendshapeTarget
{
    public string blendshapeName;
    public float targetWeight;
}


