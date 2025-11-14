using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// Expression Cue Driver - Applies micro and macro expressions from PhaseDNA
/// Reads expression cues and triggers them based on audio timeline
/// </summary>
public class ExpressionCueDriver : MonoBehaviour
{
    [Header("References")]
    public SkinnedMeshRenderer headRenderer;
    public GazeController gazeController;
    public AudioSource audioSource;
    
    [Header("Settings")]
    public float expressionIntensityMultiplier = 1f;
    public bool enableExpressions = true;
    
    // Expression cues
    private List<ExpressionCue> expressionCues = new List<ExpressionCue>();
    private int nextCueIndex = 0;
    
    // Blendshape indices (cached)
    private Dictionary<string, int> blendshapeIndices = new Dictionary<string, int>();
    
    // Active expressions
    private List<ActiveExpression> activeExpressions = new List<ActiveExpression>();
    
    // Playback state
    private bool isPlaying = false;
    private float startTime = 0f;
    private double dspStartTime = 0;
    
    // Performance
    private bool isInitialized = false;
    
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
        
        if (headRenderer != null && headRenderer.sharedMesh != null)
        {
            IndexBlendshapes();
        }
        
        if (gazeController == null)
        {
            gazeController = GetComponent<GazeController>();
        }
        
        isInitialized = true;
        Debug.Log("[Expression Cue Driver] Initialized");
    }
    
    void IndexBlendshapes()
    {
        int count = headRenderer.sharedMesh.blendShapeCount;
        
        for (int i = 0; i < count; i++)
        {
            string name = headRenderer.sharedMesh.GetBlendShapeName(i);
            string normalized = NormalizeName(name);
            
            if (!blendshapeIndices.ContainsKey(normalized))
            {
                blendshapeIndices[normalized] = i;
            }
        }
        
        Debug.Log($"[Expression Cue Driver] Indexed {blendshapeIndices.Count} blendshapes");
    }
    
    string NormalizeName(string name)
    {
        return name
            .Replace(" ", "")
            .Replace("_L", "_Left")
            .Replace("_R", "_Right")
            .ToLower();
    }
    
    void Update()
    {
        if (!isInitialized || !isPlaying || !enableExpressions) return;
        
        float currentTime = GetCurrentPlaybackTime();
        
        // Check for new cues to trigger
        while (nextCueIndex < expressionCues.Count)
        {
            ExpressionCue cue = expressionCues[nextCueIndex];
            
            if (currentTime >= cue.timestamp)
            {
                TriggerExpression(cue);
                nextCueIndex++;
            }
            else
            {
                break; // Cues are sorted by time
            }
        }
        
        // Update active expressions
        UpdateActiveExpressions(currentTime);
    }
    
    float GetCurrentPlaybackTime()
    {
        if (audioSource != null && audioSource.isPlaying)
        {
            return audioSource.time;
        }
        else if (dspStartTime > 0)
        {
            return (float)(AudioSettings.dspTime - dspStartTime);
        }
        else
        {
            return Time.time - startTime;
        }
    }
    
    void TriggerExpression(ExpressionCue cue)
    {
        Debug.Log($"[Expression] Triggering: {cue.type} (intensity: {cue.intensity})");
        
        switch (cue.type)
        {
            case ExpressionType.MicroSmile:
                ApplyMicroSmile(cue);
                break;
                
            case ExpressionType.MacroGesture:
                ApplyMacroGesture(cue);
                break;
                
            case ExpressionType.GazeShift:
                ApplyGazeShift(cue);
                break;
                
            case ExpressionType.BrowRaise:
                ApplyBrowRaise(cue);
                break;
                
            case ExpressionType.HeadNod:
                ApplyHeadNod(cue);
                break;
                
            case ExpressionType.Breath:
                ApplyBreath(cue);
                break;
        }
    }
    
    void ApplyMicroSmile(ExpressionCue cue)
    {
        float intensity = GetIntensity(cue.intensity);
        StartCoroutine(ApplyBlendshapeOverTime("mouthsmile", intensity * 30f, cue.duration));
    }
    
    void ApplyMacroGesture(ExpressionCue cue)
    {
        float intensity = GetIntensity(cue.intensity);
        
        // Combine eyebrow raise with slight head movement
        StartCoroutine(ApplyBlendshapeOverTime("browinnerup", intensity * 40f, cue.duration));
        StartCoroutine(ApplyBlendshapeOverTime("browouterup", intensity * 35f, cue.duration));
    }
    
    void ApplyGazeShift(ExpressionCue cue)
    {
        if (gazeController != null && cue.gazeTarget.HasValue)
        {
            gazeController.SetGazeTarget(cue.gazeTarget.Value);
            
            // Return to camera after duration
            StartCoroutine(ReturnGazeAfterDelay(cue.duration));
        }
    }
    
    void ApplyBrowRaise(ExpressionCue cue)
    {
        float intensity = GetIntensity(cue.intensity);
        StartCoroutine(ApplyBlendshapeOverTime("browinnerup", intensity * 50f, cue.duration));
        StartCoroutine(ApplyBlendshapeOverTime("browouterup", intensity * 45f, cue.duration));
    }
    
    void ApplyHeadNod(ExpressionCue cue)
    {
        // Animate head rotation
        StartCoroutine(AnimateHeadNod(cue.duration));
    }
    
    void ApplyBreath(ExpressionCue cue)
    {
        float intensity = GetIntensity(cue.intensity);
        StartCoroutine(ApplyBlendshapeOverTime("breathmicro", intensity * 20f, cue.duration));
    }
    
    IEnumerator ApplyBlendshapeOverTime(string blendshapeName, float targetWeight, float duration)
    {
        string normalized = NormalizeName(blendshapeName);
        
        if (!blendshapeIndices.TryGetValue(normalized, out int index))
        {
            Debug.LogWarning($"[Expression] Blendshape not found: {blendshapeName}");
            yield break;
        }
        
        float startWeight = headRenderer.GetBlendShapeWeight(index);
        float elapsed = 0f;
        float halfDuration = duration / 2f;
        
        // Ramp up
        while (elapsed < halfDuration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / halfDuration;
            float weight = Mathf.Lerp(startWeight, targetWeight * expressionIntensityMultiplier, t);
            headRenderer.SetBlendShapeWeight(index, weight);
            yield return null;
        }
        
        // Hold briefly
        headRenderer.SetBlendShapeWeight(index, targetWeight * expressionIntensityMultiplier);
        
        // Ramp down
        elapsed = 0f;
        while (elapsed < halfDuration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / halfDuration;
            float weight = Mathf.Lerp(targetWeight * expressionIntensityMultiplier, startWeight, t);
            headRenderer.SetBlendShapeWeight(index, weight);
            yield return null;
        }
        
        headRenderer.SetBlendShapeWeight(index, startWeight);
    }
    
    IEnumerator AnimateHeadNod(float duration)
    {
        Quaternion startRot = transform.localRotation;
        Quaternion downRot = startRot * Quaternion.Euler(15f, 0f, 0f);
        
        float halfDuration = duration / 2f;
        float elapsed = 0f;
        
        // Nod down
        while (elapsed < halfDuration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / halfDuration;
            transform.localRotation = Quaternion.Slerp(startRot, downRot, t);
            yield return null;
        }
        
        // Nod back up
        elapsed = 0f;
        while (elapsed < halfDuration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / halfDuration;
            transform.localRotation = Quaternion.Slerp(downRot, startRot, t);
            yield return null;
        }
        
        transform.localRotation = startRot;
    }
    
    IEnumerator ReturnGazeAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        
        if (gazeController != null)
        {
            gazeController.ResetGaze();
        }
    }
    
    void UpdateActiveExpressions(float currentTime)
    {
        // Clean up expired expressions
        activeExpressions.RemoveAll(expr => currentTime >= expr.endTime);
    }
    
    float GetIntensity(ExpressionIntensity intensity)
    {
        switch (intensity)
        {
            case ExpressionIntensity.Subtle:
                return 0.5f;
            case ExpressionIntensity.Medium:
                return 0.75f;
            case ExpressionIntensity.Emphatic:
                return 1.0f;
            default:
                return 0.75f;
        }
    }
    
    /// <summary>
    /// Load expression cues from PhaseDNA
    /// </summary>
    public void LoadExpressionCues(List<ExpressionCue> cues)
    {
        expressionCues = cues;
        expressionCues.Sort((a, b) => a.timestamp.CompareTo(b.timestamp));
        nextCueIndex = 0;
        
        Debug.Log($"[Expression Cue Driver] Loaded {expressionCues.Count} expression cues");
    }
    
    /// <summary>
    /// Start playback with audio sync
    /// </summary>
    public void StartPlayback(double dspStart = 0)
    {
        isPlaying = true;
        startTime = Time.time;
        dspStartTime = (dspStart > 0) ? dspStart : AudioSettings.dspTime;
        nextCueIndex = 0;
        
        Debug.Log("[Expression Cue Driver] Playback started");
    }
    
    /// <summary>
    /// Stop playback
    /// </summary>
    public void StopPlayback()
    {
        isPlaying = false;
        activeExpressions.Clear();
        
        Debug.Log("[Expression Cue Driver] Playback stopped");
    }
    
    /// <summary>
    /// Enable/disable expressions
    /// </summary>
    public void SetExpressionsEnabled(bool enabled)
    {
        enableExpressions = enabled;
    }
}

/// <summary>
/// Expression cue from PhaseDNA
/// </summary>
[System.Serializable]
public class ExpressionCue
{
    public string id;
    public string momentRef;
    public ExpressionType type;
    public float timestamp;
    public float offset;
    public float duration;
    public ExpressionIntensity intensity;
    public GazeTargetType? gazeTarget;
    public string notes;
}

/// <summary>
/// Expression types (from PhaseDNA schema)
/// </summary>
public enum ExpressionType
{
    MicroSmile,
    MacroGesture,
    GazeShift,
    BrowRaise,
    HeadNod,
    Breath
}

/// <summary>
/// Expression intensity levels
/// </summary>
public enum ExpressionIntensity
{
    Subtle,
    Medium,
    Emphatic
}

/// <summary>
/// Active expression tracking
/// </summary>
class ActiveExpression
{
    public ExpressionCue cue;
    public float startTime;
    public float endTime;
}



