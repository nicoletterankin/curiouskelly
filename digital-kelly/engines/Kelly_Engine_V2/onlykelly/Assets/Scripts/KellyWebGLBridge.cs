using UnityEngine;
using System.Collections;
using System.Collections.Generic;

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
    
    [Header("State")]
    public string currentExpression = "neutral";
    public bool isSpeaking = false;
    
    private Coroutine transitionCoroutine;
    private Coroutine lipSyncCoroutine;
    
    void Start()
    {
        // Auto-find components if not assigned
        if (blendshapes == null)
            blendshapes = GetComponentInChildren<ARKitBlendshapeController>();
        
        if (animator == null)
            animator = GetComponentInChildren<Animator>();
        
        Debug.Log("[KellyWebGLBridge] Ready for JavaScript commands");
        Debug.Log($"[KellyWebGLBridge] GameObject name: {gameObject.name}");
        
        // Start idle behaviors
        StartCoroutine(IdleBlink());
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











