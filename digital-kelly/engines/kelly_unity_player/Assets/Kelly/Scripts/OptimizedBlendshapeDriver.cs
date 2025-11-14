using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Optimized Blendshape Driver for 60 FPS performance
/// Enhanced version of BlendshapeDriver with performance optimizations
/// </summary>
public class OptimizedBlendshapeDriver : MonoBehaviour
{
    [Header("References")]
    public SkinnedMeshRenderer headRenderer;
    public AudioSource audioSource;
    public TextAsset a2fJsonAsset;
    
    [Header("Performance Settings")]
    public bool enableOptimizations = true;
    public float intensity = 100f;
    public int maxBlendshapesPerFrame = 20; // Limit updates per frame
    
    [Header("Blending")]
    public bool enableSmoothing = true;
    public float smoothingSpeed = 15f;
    
    // Cached data
    private Dictionary<string, int> shapeIndex = new Dictionary<string, int>();
    private A2FData data;
    private float frameTime;
    private double dspStart;
    private bool playing;
    private int blendShapeCount;
    
    // Performance optimizations
    private float[] currentWeights;
    private float[] targetWeights;
    private bool[] weightsChanged;
    private int lastFrameIndex = -1;
    
    // Statistics
    private int updatesThisFrame = 0;
    private float lastUpdateTime = 0f;
    
    void Awake()
    {
        if (headRenderer == null)
        {
            headRenderer = GetComponentInChildren<SkinnedMeshRenderer>();
        }
        
        if (a2fJsonAsset != null)
        {
            LoadRuntimeJson(a2fJsonAsset.text);
        }
        
        IndexBlendshapes();
        InitializeWeightArrays();
    }
    
    void IndexBlendshapes()
    {
        if (headRenderer == null || headRenderer.sharedMesh == null)
            return;
        
        blendShapeCount = headRenderer.sharedMesh.blendShapeCount;
        
        for (int i = 0; i < blendShapeCount; i++)
        {
            var name = headRenderer.sharedMesh.GetBlendShapeName(i);
            var norm = Normalize(name);
            if (!shapeIndex.ContainsKey(norm))
            {
                shapeIndex[norm] = i;
            }
        }
        
        Debug.Log($"[Optimized Blendshape Driver] Indexed {shapeIndex.Count} blendshapes");
    }
    
    void InitializeWeightArrays()
    {
        currentWeights = new float[blendShapeCount];
        targetWeights = new float[blendShapeCount];
        weightsChanged = new bool[blendShapeCount];
        
        // Initialize current weights from renderer
        for (int i = 0; i < blendShapeCount; i++)
        {
            currentWeights[i] = headRenderer.GetBlendShapeWeight(i);
            targetWeights[i] = 0f;
            weightsChanged[i] = false;
        }
    }
    
    string Normalize(string s) =>
        s.Replace(" ", "").Replace("_L", "_Left").Replace("_R", "_Right").ToLower();
    
    public void LoadRuntimeJson(string json)
    {
        data = JsonUtility.FromJson<A2FData>(json);
        int fps = Mathf.Max(1, data.fps);
        frameTime = 1f / fps;
        
        Debug.Log($"[Optimized Blendshape Driver] Loaded A2F data: {data.frames.Count} frames at {fps} FPS");
    }
    
    public void SetAudioClip(AudioClip clip)
    {
        if (audioSource == null)
        {
            audioSource = gameObject.GetComponent<AudioSource>() ?? gameObject.AddComponent<AudioSource>();
        }
        
        audioSource.playOnAwake = false;
        audioSource.clip = clip;
    }
    
    public void PlaySynced(double startDelay = 0.05)
    {
        if (audioSource?.clip == null || data?.frames == null || data.frames.Count == 0)
            return;
        
        dspStart = AudioSettings.dspTime + startDelay;
        audioSource.PlayScheduled(dspStart);
        playing = true;
        lastFrameIndex = -1;
        
        Debug.Log("[Optimized Blendshape Driver] Playback started");
    }
    
    void LateUpdate()
    {
        if (!playing || data == null || data.frames == null)
            return;
        
        double t = Math.Max(0, AudioSettings.dspTime - dspStart);
        int frame = (int)(t / frameTime);
        
        if (frame >= data.frames.Count)
        {
            playing = false;
            ResetBlendshapes();
            Debug.Log("[Optimized Blendshape Driver] Playback complete");
            return;
        }
        
        // Only update if frame changed (optimization)
        if (frame == lastFrameIndex && enableOptimizations)
        {
            return;
        }
        
        lastFrameIndex = frame;
        updatesThisFrame = 0;
        
        // Set target weights from A2F data
        SetTargetWeightsFromFrame(data.frames[frame]);
        
        // Apply weights with smoothing or direct
        if (enableSmoothing)
        {
            SmoothAndApplyWeights();
        }
        else
        {
            ApplyWeightsDirectly();
        }
    }
    
    void SetTargetWeightsFromFrame(Dictionary<string, float> frame)
    {
        // Clear changed flags
        Array.Clear(weightsChanged, 0, weightsChanged.Length);
        
        foreach (var kv in frame)
        {
            var key = Normalize(kv.Key);
            
            if (shapeIndex.TryGetValue(key, out int idx))
            {
                float targetWeight = Mathf.Clamp01(kv.Value) * intensity;
                
                // Only mark as changed if significantly different
                if (Mathf.Abs(targetWeights[idx] - targetWeight) > 0.1f || !enableOptimizations)
                {
                    targetWeights[idx] = targetWeight;
                    weightsChanged[idx] = true;
                }
            }
            else
            {
                // Handle aliases
                HandleBlendshapeAliases(key, kv.Value);
            }
        }
    }
    
    void HandleBlendshapeAliases(string key, float value)
    {
        // Simple aliases for common blendshape name variations
        if (key == "blinkleft" && shapeIndex.TryGetValue("eyeblink_left", out int idx))
        {
            float targetWeight = Mathf.Clamp01(value) * intensity;
            if (Mathf.Abs(targetWeights[idx] - targetWeight) > 0.1f || !enableOptimizations)
            {
                targetWeights[idx] = targetWeight;
                weightsChanged[idx] = true;
            }
        }
        
        if (key == "blinkright" && shapeIndex.TryGetValue("eyeblink_right", out idx))
        {
            float targetWeight = Mathf.Clamp01(value) * intensity;
            if (Mathf.Abs(targetWeights[idx] - targetWeight) > 0.1f || !enableOptimizations)
            {
                targetWeights[idx] = targetWeight;
                weightsChanged[idx] = true;
            }
        }
    }
    
    void SmoothAndApplyWeights()
    {
        for (int i = 0; i < blendShapeCount && updatesThisFrame < maxBlendshapesPerFrame; i++)
        {
            if (!weightsChanged[i] && enableOptimizations)
                continue;
            
            // Smooth interpolation
            currentWeights[i] = Mathf.Lerp(
                currentWeights[i],
                targetWeights[i],
                smoothingSpeed * Time.deltaTime
            );
            
            headRenderer.SetBlendShapeWeight(i, currentWeights[i]);
            updatesThisFrame++;
        }
    }
    
    void ApplyWeightsDirectly()
    {
        for (int i = 0; i < blendShapeCount && updatesThisFrame < maxBlendshapesPerFrame; i++)
        {
            if (!weightsChanged[i] && enableOptimizations)
                continue;
            
            currentWeights[i] = targetWeights[i];
            headRenderer.SetBlendShapeWeight(i, currentWeights[i]);
            updatesThisFrame++;
        }
    }
    
    void ResetBlendshapes()
    {
        for (int i = 0; i < blendShapeCount; i++)
        {
            headRenderer.SetBlendShapeWeight(i, 0f);
            currentWeights[i] = 0f;
            targetWeights[i] = 0f;
        }
    }
    
    /// <summary>
    /// Stop playback and reset
    /// </summary>
    public void Stop()
    {
        playing = false;
        
        if (audioSource != null && audioSource.isPlaying)
        {
            audioSource.Stop();
        }
        
        ResetBlendshapes();
    }
    
    /// <summary>
    /// Set intensity multiplier
    /// </summary>
    public void SetIntensity(float value)
    {
        intensity = Mathf.Clamp(value, 0f, 200f);
    }
    
    /// <summary>
    /// Enable/disable smoothing
    /// </summary>
    public void SetSmoothing(bool enabled)
    {
        enableSmoothing = enabled;
    }
    
    /// <summary>
    /// Enable/disable optimizations
    /// </summary>
    public void SetOptimizations(bool enabled)
    {
        enableOptimizations = enabled;
    }
    
    /// <summary>
    /// Get performance statistics
    /// </summary>
    public PerformanceStats GetStats()
    {
        return new PerformanceStats
        {
            totalBlendshapes = blendShapeCount,
            updatesThisFrame = updatesThisFrame,
            currentFrame = lastFrameIndex,
            totalFrames = data?.frames.Count ?? 0,
            isPlaying = playing
        };
    }
}

/// <summary>
/// Performance statistics
/// </summary>
[System.Serializable]
public struct PerformanceStats
{
    public int totalBlendshapes;
    public int updatesThisFrame;
    public int currentFrame;
    public int totalFrames;
    public bool isPlaying;
}



