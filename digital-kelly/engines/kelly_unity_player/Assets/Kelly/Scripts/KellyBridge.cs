using System.IO;
using System.Collections.Generic;
using System;
using UnityEngine;
using UnityEngine.Networking;
using System.Runtime.InteropServices;

/// <summary>
/// Kelly Bridge - Communication bridge between Flutter and Unity
/// Enhanced for Week 3: Gaze tracking, visemes, expressions, and performance
/// </summary>
public class KellyBridge : MonoBehaviour
{
#if UNITY_WEBGL && !UNITY_EDITOR
    [DllImport("__Internal")]
    private static extern void KellyPostMessage(string messageJson);

    [DllImport("__Internal")]
    private static extern void KellySubscribeToMessages(string targetObjectName, string callbackMethodName);
#endif

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
    private Coroutine activeLessonRoutine;
    private LessonRequest currentLesson;

    [System.Serializable]
    private class LessonRequest
    {
        public string lessonId;
        public string jsonUrl;
        public string audioUrl;
        public string expressionsUrl;
        public float offsetMs = 50f;
    }

    [System.Serializable]
    private class BrowserMessage
    {
        public string source;
        public string destination;
        public string type;
        public LessonRequest payload;
    }

    [System.Serializable]
    private class HostMessage
    {
        public string source = "kelly-webgl";
        public string type;
        public string status;
        public string lessonId;
        public string message;
    }

    [System.Serializable]
    private class ExpressionCueListWrapper
    {
        public List<ExpressionCue> items;
        public List<ExpressionCue> cues;
    }

    [System.Serializable]
    private class GenericListWrapper<T>
    {
        public List<T> items;
    }

    void Awake()
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        try
        {
            KellySubscribeToMessages(gameObject.name, nameof(HandleBrowserMessage));
            Debug.Log("[Kelly Bridge] WebGL message bridge registered");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[Kelly Bridge] Failed to subscribe to browser messages: {ex.Message}");
        }
#endif
    }
    
    void Start()
    {
        Initialize();
        NotifyBrowser("kelly-ready", "ok");
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
        StopPlayback();
        
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

    /// <summary>
    /// WebGL-friendly loader that accepts remote URLs via JSON payload.
    /// </summary>
    public void LoadLessonFromUrls(string payloadJson)
    {
        if (string.IsNullOrWhiteSpace(payloadJson))
        {
            Debug.LogWarning("[Kelly Bridge] LoadLessonFromUrls received empty payload");
            return;
        }

        try
        {
            var request = JsonUtility.FromJson<LessonRequest>(payloadJson);
            LoadLessonFromRequest(request, false);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[Kelly Bridge] Failed to parse remote lesson payload: {ex.Message}");
            NotifyBrowser("kelly-error", "error", null, ex.Message);
        }
    }

    /// <summary>
    /// Receives postMessage events from the browser in WebGL builds.
    /// Wired up via KellyBrowserBridge.jslib (SendMessage -> this method).
    /// </summary>
    public void HandleBrowserMessage(string payloadJson)
    {
        if (string.IsNullOrWhiteSpace(payloadJson))
        {
            return;
        }

        try
        {
            var message = JsonUtility.FromJson<BrowserMessage>(payloadJson);
            if (message == null)
            {
                Debug.LogWarning($"[Kelly Bridge] Could not deserialize browser message: {payloadJson}");
                return;
            }

            switch (message.type)
            {
                case "kelly-load":
                    if (message.payload == null)
                    {
                        NotifyBrowser("kelly-error", "error", null, "Payload missing");
                        return;
                    }
                    LoadLessonFromRequest(message.payload, true);
                    break;
                case "kelly-stop":
                    StopPlayback(true);
                    break;
                case "kelly-ping":
                    NotifyBrowser("kelly-pong", "ok");
                    break;
                default:
                    Debug.Log($"[Kelly Bridge] Unhandled browser message type: {message.type}");
                    break;
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[Kelly Bridge] Failed to handle browser message: {ex}");
            NotifyBrowser("kelly-error", "error", currentLesson?.lessonId, ex.Message);
        }
    }

    void LoadLessonFromRequest(LessonRequest request, bool notifyBrowser)
    {
        if (request == null)
        {
            Debug.LogWarning("[Kelly Bridge] LessonRequest was null");
            return;
        }

        StopPlayback();

        if (activeLessonRoutine != null)
        {
            StopCoroutine(activeLessonRoutine);
            activeLessonRoutine = null;
        }

        activeLessonRoutine = StartCoroutine(LoadLessonFromUrlsRoutine(request, notifyBrowser));
    }

    System.Collections.IEnumerator LoadLessonFromUrlsRoutine(LessonRequest request, bool notifyBrowser)
    {
        currentLesson = request;

        if (notifyBrowser)
        {
            NotifyBrowser("kelly-loading", "pending", request.lessonId);
        }

        if (string.IsNullOrWhiteSpace(request.jsonUrl) || string.IsNullOrWhiteSpace(request.audioUrl))
        {
            NotifyBrowser("kelly-error", "error", request.lessonId, "Missing jsonUrl or audioUrl");
            yield break;
        }

        // Load viseme JSON
        using (var jsonReq = UnityWebRequest.Get(request.jsonUrl))
        {
            yield return jsonReq.SendWebRequest();
            if (jsonReq.result != UnityWebRequest.Result.Success)
            {
                NotifyBrowser("kelly-error", "error", request.lessonId, jsonReq.error);
                yield break;
            }

            var json = jsonReq.downloadHandler.text;
            if (useOptimizedDriver && optimizedDriver != null)
            {
                optimizedDriver.LoadRuntimeJson(json);
            }
            else if (legacyDriver != null)
            {
                legacyDriver.LoadRuntimeJson(json);
            }
        }

        // Optional expressions
        if (!string.IsNullOrWhiteSpace(request.expressionsUrl) && expressionCueDriver != null)
        {
            using (var cueReq = UnityWebRequest.Get(request.expressionsUrl))
            {
                yield return cueReq.SendWebRequest();
                if (cueReq.result == UnityWebRequest.Result.Success)
                {
                    var cues = ParseExpressionCues(cueReq.downloadHandler.text);
                    if (cues.Count > 0)
                    {
                        expressionCueDriver.LoadExpressionCues(cues);
                    }
                }
                else
                {
                    Debug.LogWarning($"[Kelly Bridge] Failed to load expression cues: {cueReq.error}");
                }
            }
        }

        // Load audio clip
        AudioType audioType = ResolveAudioType(request.audioUrl);
        using (var audioReq = UnityWebRequestMultimedia.GetAudioClip(request.audioUrl, audioType))
        {
            yield return audioReq.SendWebRequest();
            if (audioReq.result != UnityWebRequest.Result.Success)
            {
                NotifyBrowser("kelly-error", "error", request.lessonId, audioReq.error);
                yield break;
            }

            var clip = DownloadHandlerAudioClip.GetContent(audioReq);
            double delaySeconds = Mathf.Max(0.01f, request.offsetMs / 1000f);
            StartPlaybackWithClip(clip, delaySeconds);
        }

        NotifyBrowser("kelly-playing", "ok", request.lessonId);
        activeLessonRoutine = null;
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

    private void StartPlaybackWithClip(AudioClip clip, double requestedDelaySeconds)
    {
        if (clip == null)
        {
            NotifyBrowser("kelly-error", "error", currentLesson?.lessonId, "Audio clip missing");
            return;
        }

        double delay = Math.Max(0.01, requestedDelaySeconds);
        if (audioCalibrator != null)
        {
            delay += audioCalibrator.GetOffsetSeconds();
        }

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

        if (expressionCueDriver != null)
        {
            expressionCueDriver.StartPlayback(AudioSettings.dspTime + delay);
        }
    }

    private void StopPlayback(bool notifyBrowser = false)
    {
        if (activeLessonRoutine != null)
        {
            StopCoroutine(activeLessonRoutine);
            activeLessonRoutine = null;
        }

        if (useOptimizedDriver && optimizedDriver != null)
        {
            optimizedDriver.Stop();
        }
        else if (legacyDriver != null)
        {
            legacyDriver.Stop();
        }

        if (expressionCueDriver != null)
        {
            expressionCueDriver.StopPlayback();
        }

        if (notifyBrowser)
        {
            NotifyBrowser("kelly-stopped", "ok", currentLesson?.lessonId);
        }
    }

    private AudioType ResolveAudioType(string url)
    {
        if (string.IsNullOrWhiteSpace(url))
        {
            return AudioType.WAV;
        }

        var lower = url.ToLowerInvariant();
        if (lower.EndsWith(".mp3"))
        {
            return AudioType.MPEG;
        }
        if (lower.EndsWith(".ogg"))
        {
            return AudioType.OGGVORBIS;
        }
        if (lower.EndsWith(".aac"))
        {
            return AudioType.ACC;
        }
        return AudioType.WAV;
    }

    private List<ExpressionCue> ParseExpressionCues(string json)
    {
        var cues = new List<ExpressionCue>();
        if (string.IsNullOrWhiteSpace(json))
        {
            return cues;
        }

        string trimmed = json.Trim();
        try
        {
            if (trimmed.StartsWith("["))
            {
                return ParseJsonArray<ExpressionCue>(trimmed);
            }

            var wrapper = JsonUtility.FromJson<ExpressionCueListWrapper>(trimmed);
            if (wrapper != null)
            {
                if (wrapper.items != null && wrapper.items.Count > 0)
                {
                    return wrapper.items;
                }

                if (wrapper.cues != null && wrapper.cues.Count > 0)
                {
                    return wrapper.cues;
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[Kelly Bridge] Failed to parse expression cues: {ex.Message}");
        }

        return cues;
    }

    private List<T> ParseJsonArray<T>(string json)
    {
        var wrapperJson = $"{{\"items\":{json}}}";
        var wrapper = JsonUtility.FromJson<GenericListWrapper<T>>(wrapperJson);
        return wrapper?.items ?? new List<T>();
    }

    private void NotifyBrowser(string type, string status = "ok", string lessonId = null, string message = null)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        try
        {
            var payload = new HostMessage
            {
                type = type,
                status = status,
                lessonId = lessonId,
                message = message
            };
            KellyPostMessage(JsonUtility.ToJson(payload));
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[Kelly Bridge] Failed to notify browser: {ex.Message}");
        }
#else
        if (!string.IsNullOrEmpty(message))
        {
            Debug.Log($"[Kelly Bridge] {type}: {message}");
        }
#endif
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

















