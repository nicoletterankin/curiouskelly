using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;

/// <summary>
/// WebGL Bridge for Kelly Avatar
/// Receives JavaScript SendMessage calls and controls Kelly's expressions and lip sync
/// 
/// ATTACH THIS SCRIPT TO: kelly_fbx_v4 GameObject
/// 
/// JavaScript calls from web:
///   unityInstance.SendMessage("kelly_fbx_v4", "SetExpression", "happy");
///   unityInstance.SendMessage("kelly_fbx_v4", "StartLipSync", "Hello world");
///   unityInstance.SendMessage("kelly_fbx_v4", "StopLipSync");
///   unityInstance.SendMessage("kelly_fbx_v4", "SetPhase", "welcome");
/// </summary>
public class KellyWebGLBridge : MonoBehaviour
{
    [Header("References")]
    public ARKitBlendshapeController blendshapes;
    public Animator animator;
    public AudioSource audioSource;
    public Transform headBone;

    [Header("Extracted Playback (optional)")]
    [Tooltip("If enabled, will play extracted Day 1 motion on startup (requires StreamingAssets files).")]
    public bool autoPlayExtracted = false;

    [Tooltip("StreamingAssets relative path, e.g. kelly-motion/day_001_scientist_adult_unity.json")]
    public string extractedJsonPath = "kelly-motion/day_001_scientist_adult_unity.json";

    [Tooltip("StreamingAssets relative path, e.g. kelly-motion/day_001_scientist_adult.wav")]
    public string extractedAudioPath = "kelly-motion/day_001_scientist_adult.wav";

    [Range(0f, 1f)] public float extractedVisemeStrength = 1.0f;
    [Range(0f, 1f)] public float extractedExpressionStrength = 1.0f;
    [Range(0f, 1f)] public float extractedHeadStrength = 1.0f;
    [Range(0f, 1f)] public float extractedBlinkFromEyeOpenStrength = 0.9f;
    
    [Header("State")]
    public string currentExpression = "neutral";
    public bool isSpeaking = false;
    
    private Coroutine transitionCoroutine;
    private Coroutine lipSyncCoroutine;
    private Coroutine idleBlinkCoroutine;

    // =====================================================================
    // EXTRACTED PLAYBACK DATA (from scripts/convert-to-unity.py output)
    // =====================================================================
    [Serializable]
    public class ExtractedViseme
    {
        public float time;
        public float duration;
        public string viseme;
    }

    [Serializable]
    public class ExtractedExpressionFrame
    {
        public int frame;
        public float timestamp;
        public float mouthOpen;
        public float mouthWidth;
        public float smile;
        public float leftEyeOpen;
        public float rightEyeOpen;
        public float leftBrowRaise;
        public float rightBrowRaise;
        public float headYaw;
        public float headPitch;
        public float headRoll;
    }

    [Serializable]
    public class ExtractedClip
    {
        public string clipName;
        public float duration;
        public float fps;
        public List<ExtractedViseme> visemes;
        public List<ExtractedExpressionFrame> expressions;
    }

    private ExtractedClip extracted;
    private bool extractedPlaying = false;
    private float extractedTime = 0f;
    private string activeCc5Viseme = "V_None";

    private readonly Dictionary<string, string> extractedVisemeToCc5 = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
    {
        { "viseme_sil", "V_None" },
        { "viseme_aa",  "V_Open" },
        { "viseme_PP",  "V_Explosive" },
        { "viseme_CH",  "V_Affricate" },
        { "viseme_DD",  "V_Dental_Lip" },
        { "viseme_E",   "V_Wide" },
        { "viseme_FF",  "V_Dental_Lip" },
        { "viseme_I",   "V_Wide" },
        { "viseme_O",   "V_Tight_O" },
    };

    private static readonly string[] AllCc5Visemes =
    {
        "V_None",
        "V_Open",
        "V_Explosive",
        "V_Dental_Lip",
        "V_Tight_O",
        "V_Tight",
        "V_Wide",
        "V_Affricate",
        "V_Lip_Open",
    };
    
    void Start()
    {
        // Auto-find components if not assigned
        if (blendshapes == null)
            blendshapes = GetComponentInChildren<ARKitBlendshapeController>();
        
        if (animator == null)
            animator = GetComponentInChildren<Animator>();

        if (audioSource == null)
            audioSource = GetComponent<AudioSource>();

        if (headBone == null)
            headBone = FindDeepChild(transform, "CC_Base_Head") ?? FindDeepChild(transform, "Head") ?? FindDeepChild(transform, "head");
        
        Debug.Log("[KellyWebGLBridge] Ready for JavaScript commands");
        Debug.Log($"[KellyWebGLBridge] GameObject name: {gameObject.name}");
        
        // Start idle behaviors
        idleBlinkCoroutine = StartCoroutine(IdleBlink());

        if (autoPlayExtracted)
        {
            PlayExtractedDay1();
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // JAVASCRIPT SENDMESSAGE RECEIVERS
    // These methods are called from JavaScript via unityInstance.SendMessage()
    // ═══════════════════════════════════════════════════════════════════
    
    /// <summary>
    /// Set Kelly's facial expression
    /// Called from JS: unityInstance.SendMessage("kelly_fbx_v4", "SetExpression", "happy");
    /// </summary>
    public void SetExpression(string expressionName)
    {
        Debug.Log($"[KellyWebGLBridge] SetExpression: {expressionName}");
        currentExpression = expressionName.ToLower();
        
        if (transitionCoroutine != null)
            StopCoroutine(transitionCoroutine);
        
        transitionCoroutine = StartCoroutine(TransitionToExpression(expressionName, 0.3f));
    }
    
    /// <summary>
    /// Start lip sync animation
    /// Called from JS: unityInstance.SendMessage("kelly_fbx_v4", "StartLipSync", "Hello world");
    /// Note: Web handles actual audio via ElevenLabs, this just animates mouth
    /// </summary>
    public void StartLipSync(string text)
    {
        Debug.Log($"[KellyWebGLBridge] StartLipSync: {text}");
        isSpeaking = true;
        
        if (lipSyncCoroutine != null)
            StopCoroutine(lipSyncCoroutine);
        
        lipSyncCoroutine = StartCoroutine(SimulateLipSync(text));
    }
    
    /// <summary>
    /// Stop lip sync and close mouth
    /// Called from JS: unityInstance.SendMessage("kelly_fbx_v4", "StopLipSync");
    /// </summary>
    public void StopLipSync()
    {
        Debug.Log("[KellyWebGLBridge] StopLipSync");
        isSpeaking = false;
        
        if (lipSyncCoroutine != null)
        {
            StopCoroutine(lipSyncCoroutine);
            lipSyncCoroutine = null;
        }
        
        // Close mouth
        if (blendshapes != null)
        {
            blendshapes.SetBlendshape("V_Open", 0f);
            blendshapes.SetBlendshape("Mouth_Open", 0f);
        }
    }
    
    /// <summary>
    /// Set expression based on lesson phase
    /// Called from JS: unityInstance.SendMessage("kelly_fbx_v4", "SetPhase", "welcome");
    /// </summary>
    public void SetPhase(string phaseName)
    {
        Debug.Log($"[KellyWebGLBridge] SetPhase: {phaseName}");
        
        switch (phaseName.ToLower())
        {
            case "welcome": SetExpression("curious"); break;
            case "question": 
            case "q1":
            case "q2":
            case "q3":
                SetExpression("explaining"); 
                break;
            case "wisdom": SetExpression("wisdom"); break;
            case "celebrating": SetExpression("celebrating"); break;
            default: SetExpression("neutral"); break;
        }
    }
    
    /// <summary>
    /// Set speaking state (SendMessage only accepts strings)
    /// Called from JS: unityInstance.SendMessage("kelly_fbx_v4", "SetSpeaking", "true");
    /// </summary>
    public void SetSpeaking(string speaking)
    {
        Debug.Log($"[KellyWebGLBridge] SetSpeaking: {speaking}");
        isSpeaking = speaking.ToLower() == "true";
        
        if (!isSpeaking)
            StopLipSync();
    }
    
    /// <summary>
    /// Play animation by name (if Animator is set up)
    /// Called from JS: unityInstance.SendMessage("kelly_fbx_v4", "PlayAnimation", "wave");
    /// </summary>
    public void PlayAnimation(string animationName)
    {
        Debug.Log($"[KellyWebGLBridge] PlayAnimation: {animationName}");
        
        if (animator != null)
        {
            animator.SetTrigger(animationName);
        }
    }

    // =====================================================================
    // EXTRACTED MOTION PLAYBACK (Day 1 scientist_adult)
    // =====================================================================
    public void PlayExtractedDay1()
    {
        StartCoroutine(LoadExtractedAndPlay(extractedJsonPath, extractedAudioPath));
    }

    public void StopExtracted()
    {
        extractedPlaying = false;
        extractedTime = 0f;
        extracted = null;
        activeCc5Viseme = "V_None";
        ResetAllVisemes();
        if (audioSource != null) audioSource.Stop();
        if (idleBlinkCoroutine == null) idleBlinkCoroutine = StartCoroutine(IdleBlink());
    }

    private IEnumerator LoadExtractedAndPlay(string jsonRelativePath, string audioRelativePath)
    {
        if (blendshapes == null)
        {
            Debug.LogError("[KellyWebGLBridge] Cannot play extracted: blendshapes missing.");
            yield break;
        }

        // Stop other behaviors that would fight for face control
        if (transitionCoroutine != null) StopCoroutine(transitionCoroutine);
        if (lipSyncCoroutine != null) StopCoroutine(lipSyncCoroutine);
        transitionCoroutine = null;
        lipSyncCoroutine = null;
        isSpeaking = true;

        if (idleBlinkCoroutine != null)
        {
            StopCoroutine(idleBlinkCoroutine);
            idleBlinkCoroutine = null;
        }

        // Load JSON from StreamingAssets
        string jsonUrl = CombineStreamingAssetsUrl(jsonRelativePath);
        using (var req = UnityWebRequest.Get(jsonUrl))
        {
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("[KellyWebGLBridge] Failed to load extracted JSON: " + req.error + " url=" + jsonUrl);
                yield break;
            }
            extracted = JsonUtility.FromJson<ExtractedClip>(req.downloadHandler.text);
        }

        if (extracted == null || extracted.expressions == null || extracted.expressions.Count == 0)
        {
            Debug.LogError("[KellyWebGLBridge] Extracted clip invalid (missing expressions).");
            yield break;
        }

        // Load audio (optional)
        if (audioSource != null && !string.IsNullOrWhiteSpace(audioRelativePath))
        {
            string audioUrl = CombineStreamingAssetsUrl(audioRelativePath);
            using (var req = UnityWebRequestMultimedia.GetAudioClip(audioUrl, AudioType.WAV))
            {
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                {
                    audioSource.clip = DownloadHandlerAudioClip.GetContent(req);
                }
                else
                {
                    Debug.LogWarning("[KellyWebGLBridge] Failed to load extracted audio: " + req.error + " url=" + audioUrl);
                }
            }
        }

        extractedTime = 0f;
        extractedPlaying = true;

        if (audioSource != null && audioSource.clip != null)
        {
            audioSource.time = 0f;
            audioSource.Play();
        }
    }

    private void Update()
    {
        if (!extractedPlaying || extracted == null) return;

        // Prefer audio time if we have it (better sync), else dt.
        if (audioSource != null && audioSource.isPlaying)
            extractedTime = audioSource.time;
        else
            extractedTime += Time.deltaTime;

        if (extractedTime >= extracted.duration)
        {
            StopExtracted();
            return;
        }

        ApplyExtractedAtTime(extractedTime);
    }

    private void ApplyExtractedAtTime(float t)
    {
        ApplyExtractedViseme(t);
        ApplyExtractedExpression(t);
    }

    private void ApplyExtractedViseme(float time)
    {
        if (blendshapes == null || extracted == null || extracted.visemes == null) return;

        ExtractedViseme active = null;
        for (int i = 0; i < extracted.visemes.Count; i++)
        {
            var v = extracted.visemes[i];
            if (time >= v.time && time < (v.time + v.duration))
            {
                active = v;
                break;
            }
        }

        ResetAllVisemes();
        activeCc5Viseme = "V_None";
        if (active == null || string.IsNullOrWhiteSpace(active.viseme)) return;

        if (!extractedVisemeToCc5.TryGetValue(active.viseme, out var cc5))
            cc5 = "V_None";

        activeCc5Viseme = cc5;
        if (string.Equals(activeCc5Viseme, "V_None", StringComparison.OrdinalIgnoreCase))
            return;

        blendshapes.SetBlendshape(activeCc5Viseme, Mathf.Clamp01(extractedVisemeStrength) * 100f);
    }

    private void ResetAllVisemes()
    {
        if (blendshapes == null) return;
        for (int i = 0; i < AllCc5Visemes.Length; i++)
            blendshapes.SetBlendshape(AllCc5Visemes[i], 0f);
    }

    private void ApplyExtractedExpression(float time)
    {
        if (blendshapes == null || extracted == null || extracted.expressions == null || extracted.expressions.Count == 0) return;

        int idx = Mathf.Clamp(Mathf.FloorToInt(time * extracted.fps), 0, extracted.expressions.Count - 1);
        var fr = extracted.expressions[idx];
        float e = Mathf.Clamp01(extractedExpressionStrength);

        // Smile
        float smile = Mathf.Clamp01(fr.smile) * e * 100f;
        blendshapes.SetBlendshape("Mouth_Smile_L", smile);
        blendshapes.SetBlendshape("Mouth_Smile_R", smile);

        // Brows
        float browL = Mathf.Clamp01(fr.leftBrowRaise) * e * 100f;
        float browR = Mathf.Clamp01(fr.rightBrowRaise) * e * 100f;
        blendshapes.SetBlendshape("Brow_Raise_Inner_L", browL);
        blendshapes.SetBlendshape("Brow_Raise_Inner_R", browR);

        // Eye open -> blink (inverse)
        float leftBlink = Mathf.Clamp01(1f - fr.leftEyeOpen) * extractedBlinkFromEyeOpenStrength * e * 100f;
        float rightBlink = Mathf.Clamp01(1f - fr.rightEyeOpen) * extractedBlinkFromEyeOpenStrength * e * 100f;
        blendshapes.SetBlendshape("Eye_Blink_L", leftBlink);
        blendshapes.SetBlendshape("Eye_Blink_R", rightBlink);

        // Optional mouth modifiers (avoid overriding active viseme)
        float mouthOpen = Mathf.Clamp01(fr.mouthOpen) * 35f * e;
        float mouthWidth = Mathf.Clamp01(fr.mouthWidth) * 20f * e;
        if (!string.Equals(activeCc5Viseme, "V_Open", StringComparison.OrdinalIgnoreCase))
            blendshapes.SetBlendshape("V_Open", Mathf.Max(0f, mouthOpen));
        if (!string.Equals(activeCc5Viseme, "V_Wide", StringComparison.OrdinalIgnoreCase))
            blendshapes.SetBlendshape("V_Wide", Mathf.Max(0f, mouthWidth));

        // Head
        if (headBone != null)
        {
            float h = Mathf.Clamp01(extractedHeadStrength);
            headBone.localRotation = Quaternion.Euler(fr.headPitch * h, fr.headYaw * h, fr.headRoll * h);
        }
    }

    private static string CombineStreamingAssetsUrl(string relativePath)
    {
        string basePath = Application.streamingAssetsPath;
        if (string.IsNullOrEmpty(relativePath)) return basePath;
        relativePath = relativePath.Replace("\\", "/").TrimStart('/');
        if (basePath.EndsWith("/")) return basePath + relativePath;
        return basePath + "/" + relativePath;
    }

    private static Transform FindDeepChild(Transform parent, string name)
    {
        if (parent == null) return null;
        var queue = new Queue<Transform>();
        queue.Enqueue(parent);
        while (queue.Count > 0)
        {
            var t = queue.Dequeue();
            if (string.Equals(t.name, name, StringComparison.OrdinalIgnoreCase))
                return t;
            for (int i = 0; i < t.childCount; i++)
                queue.Enqueue(t.GetChild(i));
        }
        return null;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // EXPRESSION SYSTEM
    // ═══════════════════════════════════════════════════════════════════
    
    IEnumerator TransitionToExpression(string expression, float duration)
    {
        if (blendshapes == null) yield break;
        
        // Get target weights
        var targets = GetExpressionWeights(expression.ToLower());
        
        // Smooth transition
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / duration;
            t = t * t * (3f - 2f * t); // Smoothstep
            
            // Reset and apply with lerp
            blendshapes.ResetAll();
            foreach (var kvp in targets)
            {
                blendshapes.SetBlendshape(kvp.Key, kvp.Value * t);
            }
            
            yield return null;
        }
        
        // Final values
        blendshapes.ResetAll();
        foreach (var kvp in targets)
        {
            blendshapes.SetBlendshape(kvp.Key, kvp.Value);
        }
    }
    
    Dictionary<string, float> GetExpressionWeights(string expression)
    {
        var weights = new Dictionary<string, float>();
        
        switch (expression)
        {
            case "happy":
                weights["Mouth_Smile_L"] = 70f;
                weights["Mouth_Smile_R"] = 70f;
                weights["Cheek_Raise_L"] = 40f;
                weights["Cheek_Raise_R"] = 40f;
                weights["Eye_Squint_L"] = 20f;
                weights["Eye_Squint_R"] = 20f;
                break;
                
            case "curious":
                weights["Brow_Raise_Inner_L"] = 50f;
                weights["Brow_Raise_Inner_R"] = 50f;
                weights["Eye_Wide_L"] = 25f;
                weights["Eye_Wide_R"] = 25f;
                break;
                
            case "explaining":
                weights["Brow_Raise_Outer_L"] = 35f;
                weights["Brow_Raise_Outer_R"] = 35f;
                weights["Mouth_Shrug_Upper"] = 15f;
                break;
                
            case "listening":
                weights["Brow_Raise_Inner_L"] = 25f;
                weights["Brow_Raise_Inner_R"] = 25f;
                weights["Mouth_Smile_L"] = 20f;
                weights["Mouth_Smile_R"] = 20f;
                break;
                
            case "wisdom":
                weights["Mouth_Smile_L"] = 45f;
                weights["Mouth_Smile_R"] = 45f;
                weights["Eye_Squint_L"] = 25f;
                weights["Eye_Squint_R"] = 25f;
                weights["Brow_Raise_Inner_L"] = 20f;
                weights["Brow_Raise_Inner_R"] = 20f;
                break;
                
            case "celebrating":
                weights["Mouth_Smile_L"] = 90f;
                weights["Mouth_Smile_R"] = 90f;
                weights["Cheek_Raise_L"] = 60f;
                weights["Cheek_Raise_R"] = 60f;
                weights["Eye_Squint_L"] = 35f;
                weights["Eye_Squint_R"] = 35f;
                weights["Brow_Raise_Outer_L"] = 40f;
                weights["Brow_Raise_Outer_R"] = 40f;
                break;
                
            case "neutral":
            default:
                // Empty = all zeros
                break;
        }
        
        return weights;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // LIP SYNC & IDLE BEHAVIORS
    // ═══════════════════════════════════════════════════════════════════
    
    IEnumerator SimulateLipSync(string text)
    {
        // Estimate duration based on text length (~2.5 words per second)
        float duration = text.Split(' ').Length / 2.5f;
        float elapsed = 0f;
        
        while (elapsed < duration && isSpeaking)
        {
            elapsed += Time.deltaTime;
            
            // Simple noise-based mouth movement
            float noise = Mathf.PerlinNoise(Time.time * 12f, 0f);
            float mouthOpen = noise * 50f + 10f;
            
            blendshapes?.SetBlendshape("V_Open", mouthOpen);
            
            yield return null;
        }
        
        // Close mouth
        blendshapes?.SetBlendshape("V_Open", 0f);
        isSpeaking = false;
        lipSyncCoroutine = null;
    }
    
    IEnumerator IdleBlink()
    {
        while (true)
        {
            yield return new WaitForSeconds(Random.Range(2f, 5f));
            
            if (blendshapes == null) continue;
            
            // Blink down
            for (float t = 0; t < 0.08f; t += Time.deltaTime)
            {
                float v = (t / 0.08f) * 100f;
                blendshapes.SetBlendshape("Eye_Blink_L", v);
                blendshapes.SetBlendshape("Eye_Blink_R", v);
                yield return null;
            }
            
            // Blink up
            for (float t = 0; t < 0.08f; t += Time.deltaTime)
            {
                float v = (1f - t / 0.08f) * 100f;
                blendshapes.SetBlendshape("Eye_Blink_L", v);
                blendshapes.SetBlendshape("Eye_Blink_R", v);
                yield return null;
            }
            
            blendshapes.SetBlendshape("Eye_Blink_L", 0f);
            blendshapes.SetBlendshape("Eye_Blink_R", 0f);
        }
    }
}












