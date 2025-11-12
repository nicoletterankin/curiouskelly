using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Enhanced BlendshapeDriver with 60fps support, gaze tracking, and age morphing
/// </summary>
public class BlendshapeDriver60fps : MonoBehaviour
{
    [Header("Core Components")]
    public SkinnedMeshRenderer headRenderer;
    public AudioSource audioSource;
    public TextAsset a2fJsonAsset;
    
    [Header("Animation Settings")]
    [Range(0f, 100f)] public float intensity = 100f;
    [Range(30, 120)] public int targetFPS = 60;
    public bool enableInterpolation = true;
    
    [Header("Kelly Age Settings")]
    [Range(2, 102)] public int learnerAge = 35;
    public KellyAgeVariant currentAgeVariant;
    
    [Header("Gaze Tracking")]
    public bool enableGaze = true;
    public Transform gazeTarget;
    public float gazeSpeed = 2f;
    [Range(0f, 1f)] public float gazeInfluence = 0.7f;
    public float saccadeFrequency = 3f; // per second
    
    [Header("Micro-expressions")]
    public bool enableMicroExpressions = true;
    public float blinkFrequency = 10f; // per minute
    public float breathingIntensity = 0.3f;
    
    [Header("Performance")]
    public bool showDebugInfo = false;
    
    // Private state
    private Dictionary<string, int> shapeIndex = new();
    private A2FData data;
    private float frameTime;
    private double dspStart;
    private bool playing;
    private int blendShapeCount;
    
    // 60fps interpolation
    private Dictionary<string, float> currentWeights = new();
    private Dictionary<string, float> targetWeights = new();
    
    // Gaze tracking
    private Quaternion originalHeadRotation;
    private Vector3 currentGazeOffset;
    private float nextSaccadeTime;
    
    // Micro-expressions
    private float nextBlinkTime;
    private bool isBlinking;
    private float blinkProgress;
    private float breathingPhase;
    
    // Performance tracking
    private float lastFrameTime;
    private int frameCount;
    private float fpsCounter;
    
    // Viseme-driven lip-sync (from OpenAI Realtime API)
    private Dictionary<string, float> currentVisemeWeights = new();
    private Dictionary<string, float> targetVisemeWeights = new();
    private bool usingRealtimeVisemes = false;
    private float lastVisemeUpdateTime = 0f;
    private const float VISEME_TIMEOUT = 0.2f; // Reset if no viseme update for 200ms

    void Awake()
    {
        if (headRenderer == null)
            headRenderer = GetComponentInChildren<SkinnedMeshRenderer>();
        
        if (a2fJsonAsset != null)
            LoadRuntimeJson(a2fJsonAsset.text);
        
        IndexBlendshapes();
        InitializeGaze();
        InitializeMicroExpressions();
        SetKellyAge(learnerAge);
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
                shapeIndex[norm] = i;
        }
        
        Debug.Log($"[Kelly60fps] Indexed {blendShapeCount} blendshapes");
    }

    void InitializeGaze()
    {
        if (headRenderer != null && headRenderer.transform != null)
        {
            originalHeadRotation = headRenderer.transform.localRotation;
        }
        nextSaccadeTime = Time.time + UnityEngine.Random.Range(0.2f, 0.5f);
    }

    void InitializeMicroExpressions()
    {
        nextBlinkTime = Time.time + 60f / blinkFrequency;
        breathingPhase = 0f;
    }

    string Normalize(string s) =>
        s.Replace(" ", "").Replace("_L", "_Left").Replace("_R", "_Right").ToLower();
    
    /// <summary>
    /// Mapping from OpenAI viseme names to Unity blendshape names
    /// Multiple visemes can map to the same blendshape (with different weights)
    /// </summary>
    static readonly Dictionary<string, string[]> VisemeToBlendshapeMap = new Dictionary<string, string[]>
    {
        // Vowel visemes
        { "aa", new[] { "jawOpen", "mouthOpen" } },
        { "A", new[] { "jawOpen", "mouthOpen" } },
        { "ee", new[] { "mouthSmile", "mouthWide" } },
        { "E", new[] { "mouthSmile", "mouthWide" } },
        { "ih", new[] { "mouthNarrow" } },
        { "I", new[] { "mouthNarrow" } },
        { "oh", new[] { "mouthPucker", "mouthO" } },
        { "O", new[] { "mouthPucker", "mouthO" } },
        { "ou", new[] { "mouthPucker", "mouthO" } },
        { "U", new[] { "mouthPucker", "mouthO" } },
        
        // Consonant visemes
        { "TH", new[] { "jawOpen", "tongueOut" } },
        { "FF", new[] { "mouthNarrow", "lipPress" } },
        { "DD", new[] { "tongueUp" } },
        { "SS", new[] { "mouthSmile", "tongueUp" } },
        { "NN", new[] { "tongueUp" } },
        { "RR", new[] { "tongueUp", "mouthWide" } },
        { "KK", new[] { "jawOpen", "tongueBack" } },
        { "PP", new[] { "lipPress", "mouthNarrow" } },
        { "CH", new[] { "tongueUp", "mouthPucker" } },
        
        // Silence/neutral
        { "sil", new string[0] }, // Reset blendshapes
        { "silence", new string[0] },
    };

    public void LoadRuntimeJson(string json)
    {
        data = JsonUtility.FromJson<A2FData>(json);
        int fps = data?.fps ?? targetFPS;
        frameTime = 1f / fps;
        
        Debug.Log($"[Kelly60fps] Loaded animation data: {data?.frames?.Count ?? 0} frames at {fps}fps");
    }

    public void SetAudioClip(AudioClip clip)
    {
        if (audioSource == null)
            audioSource = gameObject.GetComponent<AudioSource>() ?? gameObject.AddComponent<AudioSource>();

        audioSource.playOnAwake = false;
        audioSource.clip = clip;
    }

    public void PlaySynced(double startDelay = 0.05)
    {
        if (audioSource?.clip == null || data?.frames == null || data.frames.Count == 0)
        {
            Debug.LogWarning("[Kelly60fps] Cannot play: missing audio or animation data");
            return;
        }

        dspStart = AudioSettings.dspTime + startDelay;
        audioSource.PlayScheduled(dspStart);
        playing = true;
        
        Debug.Log($"[Kelly60fps] Started playback at 60fps");
    }

    void Update()
    {
        float deltaTime = Time.deltaTime;
        
        // Update animation
        UpdateLipSync();
        
        // Update gaze tracking
        if (enableGaze)
            UpdateGaze(deltaTime);
        
        // Update micro-expressions
        if (enableMicroExpressions)
            UpdateMicroExpressions(deltaTime);
        
        // Apply all blendshapes
        ApplyBlendshapes();
        
        // Performance tracking
        UpdatePerformanceMetrics(deltaTime);
    }

    void UpdateLipSync()
    {
        // Prioritize realtime visemes over pre-rendered A2F animation
        if (usingRealtimeVisemes)
        {
            // Viseme updates are handled in UpdateVisemes and applied in ApplyBlendshapes
            // Check for timeout
            if (Time.time - lastVisemeUpdateTime > VISEME_TIMEOUT)
            {
                usingRealtimeVisemes = false;
                targetVisemeWeights.Clear();
                Debug.Log("[Kelly60fps] Viseme stream timeout, falling back to A2F if available");
            }
            return;
        }
        
        // Fall back to A2F pre-rendered animation
        if (playing && data != null && data.frames != null)
        {
            double t = Math.Max(0, AudioSettings.dspTime - dspStart);
            int frame = (int)(t / frameTime);
            
            if (frame >= data.frames.Count)
            {
                playing = false;
                return;
            }

            var cur = data.frames[frame];

            // Update target weights
            foreach (var kv in cur)
            {
                var key = Normalize(kv.Key);
                targetWeights[key] = Mathf.Clamp01(kv.Value) * intensity;
            }

            // Smooth interpolation for 60fps
            if (enableInterpolation)
            {
                foreach (var key in targetWeights.Keys)
                {
                    if (!currentWeights.ContainsKey(key))
                        currentWeights[key] = 0f;
                    
                    // Lerp to target for smooth 60fps motion
                    float lerpSpeed = 15f; // Adjust for smoothness
                    currentWeights[key] = Mathf.Lerp(
                        currentWeights[key], 
                        targetWeights[key], 
                        lerpSpeed * Time.deltaTime
                    );
                }
            }
            else
            {
                currentWeights = new Dictionary<string, float>(targetWeights);
            }
        }
        else if (audioSource != null && audioSource.isPlaying)
        {
            // Fallback: Simple volume-based lip-sync when no A2F data or visemes
            UpdateVolumeBasedLipSync();
        }
    }
    
    /// <summary>
    /// Simple volume-based lip-sync when no A2F data is available
    /// Makes mouth open/close based on audio volume
    /// </summary>
    void UpdateVolumeBasedLipSync()
    {
        if (audioSource == null || !audioSource.isPlaying)
            return;
        
        // Get audio spectrum data
        float[] samples = new float[64];
        audioSource.GetOutputData(samples, 0);
        
        // Calculate RMS volume
        float volume = 0f;
        foreach (float sample in samples)
        {
            volume += sample * sample;
        }
        volume = Mathf.Sqrt(volume / samples.Length);
        
        // Scale volume to blendshape weight (0-100)
        float mouthOpen = Mathf.Clamp01(volume * 50f) * intensity; // Adjust multiplier for sensitivity
        
        // Apply to common mouth blendshape names
        string[] mouthBlendshapes = { "jawopen", "mouthopen", "jaw_open", "mouth_open", "jawdown", "vocalaa" };
        
        foreach (string blendName in mouthBlendshapes)
        {
            if (shapeIndex.TryGetValue(Normalize(blendName), out int idx))
            {
                currentWeights[Normalize(blendName)] = mouthOpen;
                break; // Found one, use it
            }
        }
    }

    void UpdateGaze(float deltaTime)
    {
        // Micro-saccades (small eye movements)
        if (Time.time >= nextSaccadeTime)
        {
            float range = 0.05f; // Small random movements
            currentGazeOffset = new Vector3(
                UnityEngine.Random.Range(-range, range),
                UnityEngine.Random.Range(-range, range),
                0f
            );
            
            // Next saccade in 0.25-0.5 seconds (2-4 per second)
            nextSaccadeTime = Time.time + 1f / saccadeFrequency + UnityEngine.Random.Range(0f, 0.25f);
        }

        // Apply gaze to head rotation
        if (headRenderer != null && headRenderer.transform != null)
        {
            Quaternion targetRotation = originalHeadRotation;
            
            if (gazeTarget != null)
            {
                // Look at target
                Vector3 direction = gazeTarget.position - headRenderer.transform.position;
                if (direction != Vector3.zero)
                {
                    targetRotation = Quaternion.LookRotation(direction);
                }
            }
            
            // Add micro-saccades
            Vector3 eulerOffset = currentGazeOffset * gazeInfluence * 10f;
            Quaternion saccadeRotation = Quaternion.Euler(eulerOffset);
            targetRotation = targetRotation * saccadeRotation;
            
            // Smooth rotation
            headRenderer.transform.localRotation = Quaternion.Slerp(
                headRenderer.transform.localRotation,
                targetRotation,
                gazeSpeed * deltaTime
            );
        }
    }

    void UpdateMicroExpressions(float deltaTime)
    {
        // Blinking (8-12 per minute)
        if (Time.time >= nextBlinkTime && !isBlinking)
        {
            isBlinking = true;
            blinkProgress = 0f;
        }

        if (isBlinking)
        {
            blinkProgress += deltaTime * 10f; // Blink duration ~0.2 seconds
            
            float blinkWeight = Mathf.Sin(blinkProgress * Mathf.PI) * 100f;
            
            // Apply blink to both eyes
            if (shapeIndex.TryGetValue("eyeblink_left", out int leftIdx))
                currentWeights["eyeblink_left"] = blinkWeight;
            
            if (shapeIndex.TryGetValue("eyeblink_right", out int rightIdx))
                currentWeights["eyeblink_right"] = blinkWeight;
            
            if (blinkProgress >= 1f)
            {
                isBlinking = false;
                // Schedule next blink (variation 8-12 per minute)
                float variance = UnityEngine.Random.Range(0.8f, 1.2f);
                nextBlinkTime = Time.time + (60f / blinkFrequency) * variance;
            }
        }

        // Breathing (subtle chest/shoulder movement)
        breathingPhase += deltaTime * 0.3f; // Slow breathing cycle
        float breathingWeight = Mathf.Sin(breathingPhase) * breathingIntensity * 10f;
        
        // Apply subtle breathing to relevant blendshapes
        if (shapeIndex.ContainsKey("breathing"))
        {
            currentWeights["breathing"] = breathingWeight;
        }
    }

    void ApplyBlendshapes()
    {
        if (headRenderer == null)
            return;

        // Clear all blendshapes first
        for (int i = 0; i < blendShapeCount; i++)
            headRenderer.SetBlendShapeWeight(i, 0f);

        // Apply realtime visemes if active
        if (usingRealtimeVisemes)
        {
            ApplyVisemeBlendshapes();
        }
        else
        {
            // Apply A2F animation weights
            foreach (var kv in currentWeights)
            {
                var key = kv.Key;
                var weight = kv.Value;
                
                if (shapeIndex.TryGetValue(key, out int idx))
                {
                    headRenderer.SetBlendShapeWeight(idx, weight);
                }
                else
                {
                    // Try aliases
                    if (key == "blinkleft" && shapeIndex.TryGetValue("eyeblink_left", out idx))
                        headRenderer.SetBlendShapeWeight(idx, weight);
                        
                    if (key == "blinkright" && shapeIndex.TryGetValue("eyeblink_right", out idx))
                        headRenderer.SetBlendShapeWeight(idx, weight);
                }
            }
        }
    }
    
    /// <summary>
    /// Apply viseme-driven blendshapes to avatar
    /// Maps OpenAI viseme names to Unity blendshape names and applies weights
    /// </summary>
    void ApplyVisemeBlendshapes()
    {
        if (headRenderer == null)
            return;
        
        // Interpolate current viseme weights to target for smooth 60fps transitions
        float lerpSpeed = 20f; // Fast interpolation for real-time responsiveness
        
        foreach (var visemeEntry in targetVisemeWeights)
        {
            string visemeName = visemeEntry.Key.ToLower();
            float targetWeight = Mathf.Clamp01(visemeEntry.Value);
            
            // Handle silence - reset all viseme-related blendshapes
            if (visemeName == "sil" || visemeName == "silence")
            {
                // Don't apply silence viseme, it's a signal to reset
                continue;
            }
            
            // Get mapped blendshape names for this viseme
            if (!VisemeToBlendshapeMap.TryGetValue(visemeName, out string[] blendshapeNames) || blendshapeNames.Length == 0)
            {
                // Try case-insensitive lookup
                var found = false;
                foreach (var kv in VisemeToBlendshapeMap)
                {
                    if (string.Equals(kv.Key, visemeName, StringComparison.OrdinalIgnoreCase))
                    {
                        blendshapeNames = kv.Value;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    // Unknown viseme, skip
                    continue;
                }
            }
            
            // Apply weight to each mapped blendshape
            foreach (string blendshapeName in blendshapeNames)
            {
                string normalizedName = Normalize(blendshapeName);
                
                // Interpolate current weight to target
                if (!currentVisemeWeights.ContainsKey(normalizedName))
                    currentVisemeWeights[normalizedName] = 0f;
                
                currentVisemeWeights[normalizedName] = Mathf.Lerp(
                    currentVisemeWeights[normalizedName],
                    targetWeight * intensity,
                    lerpSpeed * Time.deltaTime
                );
                
                // Apply to blendshape
                if (shapeIndex.TryGetValue(normalizedName, out int idx))
                {
                    headRenderer.SetBlendShapeWeight(idx, currentVisemeWeights[normalizedName]);
                }
            }
        }
        
        // Build set of all blendshapes that should be active from current visemes
        var activeBlendshapes = new HashSet<string>();
        foreach (var visemeEntry in targetVisemeWeights)
        {
            string visemeName = visemeEntry.Key.ToLower();
            if (visemeName == "sil" || visemeName == "silence")
                continue;
            
            if (VisemeToBlendshapeMap.TryGetValue(visemeName, out string[] blendshapeNames))
            {
                foreach (string blendshapeName in blendshapeNames)
                {
                    activeBlendshapes.Add(Normalize(blendshapeName));
                }
            }
        }
        
        // Fade out blendshapes that are no longer active
        var toRemove = new List<string>();
        foreach (var kv in currentVisemeWeights)
        {
            if (!activeBlendshapes.Contains(kv.Key))
            {
                // Fade out
                float newWeight = Mathf.Lerp(kv.Value, 0f, lerpSpeed * Time.deltaTime);
                if (newWeight < 0.01f)
                {
                    toRemove.Add(kv.Key);
                }
                else
                {
                    currentVisemeWeights[kv.Key] = newWeight;
                    if (shapeIndex.TryGetValue(kv.Key, out int idx))
                    {
                        headRenderer.SetBlendShapeWeight(idx, newWeight);
                    }
                }
            }
        }
        
        foreach (var key in toRemove)
        {
            currentVisemeWeights.Remove(key);
        }
    }
    
    /// <summary>
    /// Update visemes from OpenAI Realtime API
    /// Called by KellyAvatarController when viseme data arrives from Flutter
    /// </summary>
    public void UpdateVisemes(Dictionary<string, float> visemes)
    {
        if (visemes == null || visemes.Count == 0)
        {
            // No visemes - reset
            if (usingRealtimeVisemes)
            {
                targetVisemeWeights.Clear();
                usingRealtimeVisemes = false;
            }
            return;
        }
        
        // Update target viseme weights
        targetVisemeWeights.Clear();
        foreach (var kv in visemes)
        {
            targetVisemeWeights[kv.Key.ToLower()] = Mathf.Clamp01(kv.Value);
        }
        
        // Enable realtime viseme mode
        usingRealtimeVisemes = true;
        lastVisemeUpdateTime = Time.time;
        
        // If A2F animation is playing, stop it (realtime visemes take priority)
        if (playing)
        {
            playing = false;
            if (audioSource != null)
            {
                audioSource.Stop();
            }
        }
    }

    void UpdatePerformanceMetrics(float deltaTime)
    {
        frameCount++;
        fpsCounter += deltaTime;
        
        if (fpsCounter >= 1f)
        {
            float currentFPS = frameCount / fpsCounter;
            
            if (showDebugInfo)
            {
                Debug.Log($"[Kelly60fps] FPS: {currentFPS:F1}, " +
                         $"Blendshapes: {currentWeights.Count}, " +
                         $"Playing: {playing}, " +
                         $"Age: {GetKellyAge()}");
            }
            
            frameCount = 0;
            fpsCounter = 0f;
        }
        
        lastFrameTime = deltaTime;
    }

    /// <summary>
    /// Set Kelly's age based on learner age
    /// </summary>
    public void SetKellyAge(int age)
    {
        learnerAge = Mathf.Clamp(age, 2, 102);
        
        // Determine Kelly's age
        int kellyAge;
        if (learnerAge <= 5) kellyAge = 3;
        else if (learnerAge <= 12) kellyAge = 9;
        else if (learnerAge <= 17) kellyAge = 15;
        else if (learnerAge <= 35) kellyAge = 27;
        else if (learnerAge <= 60) kellyAge = 48;
        else kellyAge = 82;
        
        // Update age variant
        currentAgeVariant = (KellyAgeVariant)kellyAge;
        
        // Adjust micro-expression frequencies based on age
        if (kellyAge <= 9)
        {
            blinkFrequency = 15f; // More frequent for kids
            saccadeFrequency = 4f; // More energetic movements
        }
        else if (kellyAge >= 48)
        {
            blinkFrequency = 8f; // Less frequent for elders
            saccadeFrequency = 2f; // Slower, more deliberate
        }
        else
        {
            blinkFrequency = 10f; // Default
            saccadeFrequency = 3f;
        }
        
        Debug.Log($"[Kelly60fps] Set Kelly age to {kellyAge} for learner age {learnerAge}");
    }

    public int GetKellyAge()
    {
        if (learnerAge <= 5) return 3;
        if (learnerAge <= 12) return 9;
        if (learnerAge <= 17) return 15;
        if (learnerAge <= 35) return 27;
        if (learnerAge <= 60) return 48;
        return 82;
    }

    void OnGUI()
    {
        if (!showDebugInfo)
            return;

        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label($"Kelly Avatar - 60fps");
        GUILayout.Label($"FPS: {(1f / lastFrameTime):F1}");
        GUILayout.Label($"Learner Age: {learnerAge}");
        GUILayout.Label($"Kelly Age: {GetKellyAge()}");
        GUILayout.Label($"Active Blendshapes: {currentWeights.Count}");
        GUILayout.Label($"Playing: {(playing ? "Yes" : "No")}");
        GUILayout.Label($"Gaze: {(enableGaze ? "On" : "Off")}");
        GUILayout.Label($"Micro-expressions: {(enableMicroExpressions ? "On" : "Off")}");
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
}

/// <summary>
/// Kelly's age variants for visual morphing
/// </summary>
public enum KellyAgeVariant
{
    Toddler = 3,
    Kid = 9,
    Teen = 15,
    Adult = 27,
    Mentor = 48,
    Elder = 82
}

// A2FData class is defined in A2FModels.cs - removed duplicate definition here



