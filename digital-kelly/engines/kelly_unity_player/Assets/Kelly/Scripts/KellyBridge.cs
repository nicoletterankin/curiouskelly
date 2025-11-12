using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Kelly Bridge - Communication bridge between Flutter and Unity
/// Enhanced for Week 3: Gaze tracking, visemes, expressions, and performance
/// </summary>
public class KellyBridge : MonoBehaviour
{
    [Header("Legacy Driver (Week 2)")]
    public BlendshapeDriver legacyDriver;
    
    [Header("Week 3: Optimized Systems")]
    public OptimizedBlendshapeDriver optimizedDriver;
    public VisemeMapper visemeMapper;
    public GazeController gazeController;
    public ExpressionCueDriver expressionCueDriver;
    public AudioSyncCalibrator audioCalibrator;
    
    [Header("Performance Monitoring")]
    public FPSCounter fpsCounter;
    public PerformanceMonitor performanceMonitor;
    
    [Header("Settings")]
    public bool useOptimizedDriver = true;
    
    private bool isInitialized = false;

    void Start()
    {
        Initialize();
    }
    
    void Initialize()
    {
        if (isInitialized) return;
        
        // Auto-find components if not set
        if (optimizedDriver == null)
            optimizedDriver = GetComponent<OptimizedBlendshapeDriver>();
        
        if (visemeMapper == null)
            visemeMapper = GetComponent<VisemeMapper>();
        
        if (gazeController == null)
            gazeController = GetComponent<GazeController>();
        
        if (expressionCueDriver == null)
            expressionCueDriver = GetComponent<ExpressionCueDriver>();
        
        if (audioCalibrator == null)
            audioCalibrator = GetComponent<AudioSyncCalibrator>();
        
        if (fpsCounter == null)
            fpsCounter = FindObjectOfType<FPSCounter>();
        
        if (performanceMonitor == null)
            performanceMonitor = FindObjectOfType<PerformanceMonitor>();
        
        isInitialized = true;
        Debug.Log("[Kelly Bridge] Initialized (Week 3 Enhanced)");
    }

    // ===== LEGACY SUPPORT (Week 2) =====
    
    /// <summary>
    /// Legacy method: Load and play (Week 2 compatibility)
    /// Called by Flutter: payload = "path/to/json|path/to/wav"
    /// </summary>
    public void LoadAndPlay(string payload)
    {
        var parts = payload.Split('|');
        var jsonPath = parts[0];
        var wavPath = parts[1];

        Debug.Log($"📥 KellyBridge: Received load request - JSON: {jsonPath}, WAV: {wavPath}");

        if (File.Exists(jsonPath))
        {
            string json = File.ReadAllText(jsonPath);
            
            // Use optimized driver if enabled
            if (useOptimizedDriver && optimizedDriver != null)
            {
                optimizedDriver.LoadRuntimeJson(json);
            }
            else if (legacyDriver != null)
            {
                legacyDriver.LoadRuntimeJson(json);
            }
            
            Debug.Log("✅ KellyBridge: Loaded A2F data");
        }
        else
        {
            Debug.LogWarning($"⚠️ KellyBridge: JSON file not found: {jsonPath}");
        }

        if (File.Exists(wavPath))
        {
            StartCoroutine(LoadClipAndPlay(wavPath));
        }
        else
        {
            Debug.LogWarning($"⚠️ KellyBridge: WAV file not found: {wavPath}");
        }
    }

    private System.Collections.IEnumerator LoadClipAndPlay(string path)
    {
        Debug.Log($"🎵 KellyBridge: Loading audio from {path}");
        using var req = UnityWebRequestMultimedia.GetAudioClip("file://" + path, AudioType.WAV);
        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
        {
            var clip = DownloadHandlerAudioClip.GetContent(req);
            
            // Apply audio sync calibration
            double delay = 0.05;
            if (audioCalibrator != null)
            {
                delay += audioCalibrator.GetOffsetSeconds();
            }
            
            // Use optimized driver if enabled
            if (useOptimizedDriver && optimizedDriver != null)
            {
                optimizedDriver.SetAudioClip(clip);
                optimizedDriver.PlaySynced(delay);
            }
            else if (legacyDriver != null)
            {
                legacyDriver.SetAudioClip(clip);
                legacyDriver.PlaySynced(delay);
            }
            
            // Start expression cues if available
            if (expressionCueDriver != null)
            {
                expressionCueDriver.StartPlayback(AudioSettings.dspTime + delay);
            }
            
            Debug.Log("✅ KellyBridge: Audio playing in sync");
        }
        else
        {
            Debug.LogError($"❌ KellyBridge: Failed to load audio: {req.error}");
        }
    }
    
    // ===== WEEK 3: NEW METHODS =====
    
    /// <summary>
    /// Apply single viseme from Flutter/OpenAI Realtime API
    /// </summary>
    public void ApplyViseme(string visemeId, float weight = 1f)
    {
        if (visemeMapper != null)
        {
            visemeMapper.ApplyViseme(visemeId, weight);
        }
    }
    
    /// <summary>
    /// Apply multiple visemes with blending (JSON format)
    /// Example: {"aa": 0.8, "E": 0.2}
    /// </summary>
    public void ApplyVisemes(string visemesJson)
    {
        if (visemeMapper != null)
        {
            var visemes = JsonUtility.FromJson<Dictionary<string, float>>(visemesJson);
            visemeMapper.ApplyVisemes(visemes);
        }
    }
    
    /// <summary>
    /// Set gaze target by type
    /// </summary>
    public void SetGazeTarget(string targetType)
    {
        if (gazeController == null) return;
        
        switch (targetType.ToLower())
        {
            case "camera":
                gazeController.SetGazeTarget(GazeTargetType.Camera);
                break;
            case "left":
                gazeController.SetGazeTarget(GazeTargetType.Left);
                break;
            case "right":
                gazeController.SetGazeTarget(GazeTargetType.Right);
                break;
            case "up":
                gazeController.SetGazeTarget(GazeTargetType.Up);
                break;
            case "down":
                gazeController.SetGazeTarget(GazeTargetType.Down);
                break;
            case "content":
                gazeController.SetGazeTarget(GazeTargetType.Content);
                break;
        }
    }
    
    /// <summary>
    /// Set gaze target from screen position (for touch interaction)
    /// </summary>
    public void SetGazeFromScreen(float x, float y)
    {
        if (gazeController != null)
        {
            gazeController.SetGazeTargetFromScreen(new Vector2(x, y));
        }
    }
    
    /// <summary>
    /// Load expression cues from JSON
    /// </summary>
    public void LoadExpressionCues(string cuesJson)
    {
        if (expressionCueDriver != null)
        {
            var cues = JsonUtility.FromJson<List<ExpressionCue>>(cuesJson);
            expressionCueDriver.LoadExpressionCues(cues);
        }
    }
    
    /// <summary>
    /// Set audio sync calibration offset (milliseconds)
    /// </summary>
    public void SetAudioOffset(float offsetMs)
    {
        if (audioCalibrator != null)
        {
            audioCalibrator.SetOffset(offsetMs);
        }
    }
    
    /// <summary>
    /// Play calibration test
    /// </summary>
    public void PlayCalibrationTest()
    {
        if (audioCalibrator != null)
        {
            audioCalibrator.PlayTestAudio();
        }
    }
    
    /// <summary>
    /// Save audio calibration
    /// </summary>
    public void SaveCalibration()
    {
        if (audioCalibrator != null)
        {
            audioCalibrator.SaveCalibration();
        }
    }
    
    /// <summary>
    /// Get performance metrics (returns JSON string)
    /// </summary>
    public string GetPerformanceMetrics()
    {
        if (performanceMonitor != null)
        {
            return performanceMonitor.ExportToJson();
        }
        return "{}";
    }
    
    /// <summary>
    /// Get current FPS
    /// </summary>
    public float GetCurrentFPS()
    {
        return fpsCounter != null ? fpsCounter.GetCurrentFPS() : 0f;
    }
    
    /// <summary>
    /// Enable/disable micro-saccades
    /// </summary>
    public void SetMicroSaccadesEnabled(bool enabled)
    {
        if (gazeController != null)
        {
            gazeController.SetMicroSaccadesEnabled(enabled);
        }
    }
    
    /// <summary>
    /// Enable/disable expressions
    /// </summary>
    public void SetExpressionsEnabled(bool enabled)
    {
        if (expressionCueDriver != null)
        {
            expressionCueDriver.SetExpressionsEnabled(enabled);
        }
    }
    
    /// <summary>
    /// Switch between optimized and legacy driver
    /// </summary>
    public void SetOptimizedDriver(bool enabled)
    {
        useOptimizedDriver = enabled;
        Debug.Log($"[Kelly Bridge] Using {(enabled ? "Optimized" : "Legacy")} driver");
    }
}

















