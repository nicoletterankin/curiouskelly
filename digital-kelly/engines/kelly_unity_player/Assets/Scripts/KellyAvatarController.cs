using System;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;

/// <summary>
/// Main controller for Kelly avatar
/// Handles age morphing, voice sync, and Flutter communication
/// </summary>
public class KellyAvatarController : MonoBehaviour
{
    [Header("Components")]
    public BlendshapeDriver60fps blendshapeDriver;
    public AvatarPerformanceMonitor performanceMonitor;
    public GameObject[] kellyAgeVariants; // 6 models: ages 3, 9, 15, 27, 48, 82
    
    [Header("Current State")]
    public int currentLearnerAge = 35;
    public int currentKellyAge = 27;
    public string currentTopic = "leaves";
    
    [Header("Voice Settings")]
    public float voicePitch = 1.0f;
    public float voiceSpeed = 1.0f;
    
    [Header("Status")]
    public bool isReady = false;
    public bool isPlaying = false;
    public bool isSpeaking = false;

    void Start()
    {
        InitializeAvatar();
        
        // Subscribe to messages from Flutter (if UnityMessageManager exists)
        if (UnityMessageManager.Instance != null)
        {
            UnityMessageManager.Instance.OnMessage += HandleFlutterMessage;
        }
        else
        {
            Debug.LogWarning("[KellyController] UnityMessageManager not found - Flutter communication disabled. This is OK for standalone Unity testing.");
        }
        
        SendReadyMessage();
    }

    void InitializeAvatar()
    {
        // Auto-find components if not assigned
        if (blendshapeDriver == null)
            blendshapeDriver = GetComponent<BlendshapeDriver60fps>();
        
        if (performanceMonitor == null)
            performanceMonitor = GetComponent<AvatarPerformanceMonitor>();
        
        // Set initial age
        SetLearnerAge(currentLearnerAge);
        
        isReady = true;
        Debug.Log("[KellyController] Avatar initialized and ready");
    }

    void HandleFlutterMessage(string message)
    {
        try
        {
            // Check if this is an updateVisemes message (needs special parsing for nested JSON)
            if (message.Contains("\"type\":\"updateVisemes\""))
            {
                var visemeMap = ParseVisemeJsonFromMessage(message);
                if (visemeMap != null && blendshapeDriver != null)
                {
                    blendshapeDriver.UpdateVisemes(visemeMap);
                }
                return;
            }
            
            var msg = JsonUtility.FromJson<FlutterMessage>(message);
            
            switch (msg.type)
            {
                case "setAge":
                    SetLearnerAge(msg.age);
                    break;
                    
                case "playLesson":
                    PlayLesson(msg.lessonId, msg.age);
                    break;
                    
                case "speak":
                    Speak(msg.text, msg.age);
                    break;
                    
                case "stop":
                    StopPlayback();
                    break;
                    
                case "getPerformance":
                    SendPerformanceStats();
                    break;
                    
                case "setGazeTarget":
                    SetGazeTarget(msg.x, msg.y);
                    break;
                
                default:
                    Debug.LogWarning($"[KellyController] Unknown message type: {msg.type}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[KellyController] Error handling message: {e.Message}");
        }
    }

    public void SetLearnerAge(int age)
    {
        currentLearnerAge = Mathf.Clamp(age, 2, 102);
        currentKellyAge = GetKellyAge(currentLearnerAge);
        
        // Update blendshape driver
        if (blendshapeDriver != null)
        {
            blendshapeDriver.SetKellyAge(currentLearnerAge);
        }
        
        // Switch to appropriate Kelly age variant model (if available)
        SwitchAgeVariant(currentKellyAge);
        
        // Update voice parameters based on age
        UpdateVoiceParameters(currentKellyAge);
        
        // Send update to Flutter
        SendToFlutter("ageUpdated", new
        {
            learnerAge = currentLearnerAge,
            kellyAge = currentKellyAge,
            voicePitch = voicePitch,
            voiceSpeed = voiceSpeed
        });
        
        Debug.Log($"[KellyController] Set learner age to {currentLearnerAge}, Kelly is now {currentKellyAge}");
    }

    int GetKellyAge(int learnerAge)
    {
        if (learnerAge <= 5) return 3;
        if (learnerAge <= 12) return 9;
        if (learnerAge <= 17) return 15;
        if (learnerAge <= 35) return 27;
        if (learnerAge <= 60) return 48;
        return 82;
    }

    void SwitchAgeVariant(int kellyAge)
    {
        if (kellyAgeVariants == null || kellyAgeVariants.Length == 0)
            return;

        // Map Kelly age to variant index
        int variantIndex = kellyAge switch
        {
            3 => 0,
            9 => 1,
            15 => 2,
            27 => 3,
            48 => 4,
            82 => 5,
            _ => 3 // Default to adult
        };

        // Activate correct variant, deactivate others
        for (int i = 0; i < kellyAgeVariants.Length; i++)
        {
            if (kellyAgeVariants[i] != null)
            {
                kellyAgeVariants[i].SetActive(i == variantIndex);
            }
        }
        
        Debug.Log($"[KellyController] Switched to age variant {kellyAge} (index {variantIndex})");
    }

    void UpdateVoiceParameters(int kellyAge)
    {
        // Voice characteristics for each Kelly age
        switch (kellyAge)
        {
            case 3: // Toddler
                voicePitch = 1.3f;
                voiceSpeed = 0.9f;
                break;
            case 9: // Kid
                voicePitch = 1.15f;
                voiceSpeed = 1.0f;
                break;
            case 15: // Teen
                voicePitch = 1.05f;
                voiceSpeed = 1.1f;
                break;
            case 27: // Adult
                voicePitch = 1.0f;
                voiceSpeed = 1.0f;
                break;
            case 48: // Mentor
                voicePitch = 0.95f;
                voiceSpeed = 0.95f;
                break;
            case 82: // Elder
                voicePitch = 0.9f;
                voiceSpeed = 0.85f;
                break;
        }
    }

    public void PlayLesson(string lessonId, int age)
    {
        currentLearnerAge = age;
        currentTopic = lessonId;
        
        SetLearnerAge(age);
        
        Debug.Log($"[KellyController] Playing lesson: {lessonId} for age {age}");
        
        // Notify Flutter
        SendToFlutter("lessonStarted", new
        {
            lessonId = lessonId,
            age = age,
            kellyAge = currentKellyAge
        });
        
        isPlaying = true;
    }

    public void Speak(string text, int age)
    {
        SetLearnerAge(age);
        
        Debug.Log($"[KellyController] Speaking: {text.Substring(0, Math.Min(50, text.Length))}...");
        
        isSpeaking = true;
        
        // Notify Flutter
        SendToFlutter("speaking", new
        {
            text = text,
            kellyAge = currentKellyAge
        });
    }

    public void StopPlayback()
    {
        if (blendshapeDriver != null && blendshapeDriver.audioSource != null)
        {
            blendshapeDriver.audioSource.Stop();
        }
        
        isPlaying = false;
        isSpeaking = false;
        
        Debug.Log("[KellyController] Playback stopped");
        
        SendToFlutter("stopped", null);
    }

    public void SetGazeTarget(float x, float y)
    {
        if (blendshapeDriver != null && blendshapeDriver.gazeTarget != null)
        {
            // Convert screen coordinates to world position
            Vector3 screenPos = new Vector3(x * Screen.width, y * Screen.height, 2f);
            Vector3 worldPos = Camera.main.ScreenToWorldPoint(screenPos);
            blendshapeDriver.gazeTarget.position = worldPos;
        }
    }

    void SendPerformanceStats()
    {
        if (performanceMonitor == null)
            return;

        var stats = performanceMonitor.GetCurrentStats();
        
        SendToFlutter("performanceStats", new
        {
            fps = stats.currentFPS,
            avgFps = stats.averageFPS,
            minFps = stats.minFPS,
            maxFps = stats.maxFPS,
            frameTimeMs = stats.frameTimeMs,
            memoryMB = stats.unityMemoryMB,
            meetingTarget = stats.meetingTarget,
            status = stats.status
        });
    }

    void SendReadyMessage()
    {
        SendToFlutter("ready", new
        {
            kellyAge = currentKellyAge,
            learnerAge = currentLearnerAge,
            fps = Application.targetFrameRate,
            platform = Application.platform.ToString()
        });
    }

    void SendToFlutter(string type, object data)
    {
        try
        {
            if (UnityMessageManager.Instance == null)
                return; // Skip if no UnityMessageManager (standalone Unity mode)
            
            var message = new
            {
                type = type,
                data = data,
                timestamp = DateTime.UtcNow.ToString("o")
            };
            
            string json = JsonUtility.ToJson(message);
            UnityMessageManager.Instance.SendMessageToFlutter(json);
        }
        catch (System.Exception ex)
        {
            // Silently ignore Flutter communication errors in standalone Unity mode
            Debug.LogWarning($"[KellyController] Flutter communication unavailable: {ex.Message}");
        }
    }

    /// <summary>
    /// Parse viseme JSON from full Flutter message
    /// Handles JSON like: {"type":"updateVisemes","visemes":{"aa":0.5,"ee":0.3,"ih":0.2}}
    /// </summary>
    Dictionary<string, float> ParseVisemeJsonFromMessage(string messageJson)
    {
        var result = new Dictionary<string, float>();
        
        if (string.IsNullOrEmpty(messageJson))
            return result;
        
        try
        {
            // Find the start of the visemes object
            int visemesStart = messageJson.IndexOf("\"visemes\":");
            if (visemesStart == -1)
            {
                Debug.LogWarning("[KellyController] Could not find visemes field in message");
                return result;
            }
            
            // Find the opening brace after "visemes":
            int objectStart = messageJson.IndexOf('{', visemesStart);
            if (objectStart == -1)
            {
                Debug.LogWarning("[KellyController] Could not find visemes object start");
                return result;
            }
            
            // Find the matching closing brace (handle nested objects)
            int braceCount = 0;
            int objectEnd = -1;
            for (int i = objectStart; i < messageJson.Length; i++)
            {
                if (messageJson[i] == '{')
                    braceCount++;
                else if (messageJson[i] == '}')
                {
                    braceCount--;
                    if (braceCount == 0)
                    {
                        objectEnd = i;
                        break;
                    }
                }
            }
            
            if (objectEnd == -1)
            {
                Debug.LogWarning("[KellyController] Could not find visemes object end");
                return result;
            }
            
            // Extract the visemes object content (without outer braces)
            string visemeContent = messageJson.Substring(objectStart + 1, objectEnd - objectStart - 1);
            
            // Parse key-value pairs from visemes object
            // Pattern: "key":value (handles decimal numbers)
            var matches = Regex.Matches(visemeContent, @"\""([^\""]+)\"":\s*([0-9]*\.?[0-9]+)");
            
            foreach (Match match in matches)
            {
                if (match.Groups.Count >= 3)
                {
                    string key = match.Groups[1].Value;
                    if (float.TryParse(match.Groups[2].Value, System.Globalization.NumberStyles.Float, 
                        System.Globalization.CultureInfo.InvariantCulture, out float value))
                    {
                        result[key] = value;
                    }
                }
            }
            
            if (result.Count > 0)
            {
                Debug.Log($"[KellyController] Parsed {result.Count} visemes from message");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[KellyController] Error parsing viseme JSON: {e.Message}");
        }
        
        return result;
    }

    void OnDestroy()
    {
        // Cleanup
        if (UnityMessageManager.Instance != null)
        {
            UnityMessageManager.Instance.OnMessage -= HandleFlutterMessage;
        }
    }

    // For testing in Unity Editor
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 220, 300, 300));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label("Kelly Controller", GUI.skin.GetStyle("boldLabel"));
        GUILayout.Space(5);
        
        GUILayout.Label($"Learner Age: {currentLearnerAge}");
        GUILayout.Label($"Kelly Age: {currentKellyAge}");
        GUILayout.Label($"Topic: {currentTopic}");
        GUILayout.Label($"Voice Pitch: {voicePitch:F2}x");
        GUILayout.Label($"Voice Speed: {voiceSpeed:F2}x");
        GUILayout.Label($"Ready: {(isReady ? "Yes" : "No")}");
        GUILayout.Label($"Playing: {(isPlaying ? "Yes" : "No")}");
        GUILayout.Label($"Speaking: {(isSpeaking ? "Yes" : "No")}");
        
        GUILayout.Space(10);
        
        // Test controls
        if (GUILayout.Button("Test Age 5"))
            SetLearnerAge(5);
        
        if (GUILayout.Button("Test Age 35"))
            SetLearnerAge(35);
        
        if (GUILayout.Button("Test Age 102"))
            SetLearnerAge(102);
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
}

/// <summary>
/// Message structure from Flutter
/// </summary>
[Serializable]
public class FlutterMessage
{
    public string type;
    public int age;
    public string lessonId;
    public string text;
    public float x;
    public float y;
    public string visemes; // JSON string representation of viseme map
}



