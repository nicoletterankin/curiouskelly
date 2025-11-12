using UnityEngine;
using System.Collections;

/// <summary>
/// Audio Sync Calibrator - Adjusts audio/visual sync offset
/// Allows per-device calibration for frame-accurate lip-sync
/// Target: <5% lip-sync error
/// </summary>
public class AudioSyncCalibrator : MonoBehaviour
{
    [Header("Calibration Settings")]
    [Range(-60f, 60f)]
    public float offsetMs = 0f; // Milliseconds offset
    
    [Header("Testing")]
    public AudioClip testAudioClip;
    public TextAsset testA2FJson;
    
    [Header("References")]
    public OptimizedBlendshapeDriver blendshapeDriver;
    public AudioSource audioSource;
    
    // Calibration state
    private bool isCalibrating = false;
    private float savedOffset = 0f;
    
    // Persistence
    private const string PREF_KEY = "AudioSyncOffset";
    private const string PREF_DEVICE_KEY = "AudioSyncDevice";
    
    void Start()
    {
        // Load saved offset for this device
        LoadCalibration();
        
        if (blendshapeDriver == null)
        {
            blendshapeDriver = GetComponent<OptimizedBlendshapeDriver>();
        }
        
        if (audioSource == null)
        {
            audioSource = GetComponent<AudioSource>();
        }
        
        Debug.Log($"[Audio Sync Calibrator] Initialized with offset: {offsetMs}ms");
    }
    
    /// <summary>
    /// Load calibration for current device
    /// </summary>
    void LoadCalibration()
    {
        string deviceId = GetDeviceIdentifier();
        string savedDevice = PlayerPrefs.GetString(PREF_DEVICE_KEY, "");
        
        if (savedDevice == deviceId)
        {
            offsetMs = PlayerPrefs.GetFloat(PREF_KEY, 0f);
            Debug.Log($"[Audio Sync Calibrator] Loaded offset for device {deviceId}: {offsetMs}ms");
        }
        else
        {
            Debug.Log($"[Audio Sync Calibrator] New device detected: {deviceId}");
            offsetMs = 0f;
        }
    }
    
    /// <summary>
    /// Save calibration for current device
    /// </summary>
    public void SaveCalibration()
    {
        string deviceId = GetDeviceIdentifier();
        PlayerPrefs.SetString(PREF_DEVICE_KEY, deviceId);
        PlayerPrefs.SetFloat(PREF_KEY, offsetMs);
        PlayerPrefs.Save();
        
        Debug.Log($"[Audio Sync Calibrator] Saved offset for device {deviceId}: {offsetMs}ms");
    }
    
    /// <summary>
    /// Get unique device identifier
    /// </summary>
    string GetDeviceIdentifier()
    {
        return $"{SystemInfo.deviceModel}_{SystemInfo.operatingSystem}";
    }
    
    /// <summary>
    /// Set calibration offset
    /// </summary>
    public void SetOffset(float ms)
    {
        offsetMs = Mathf.Clamp(ms, -60f, 60f);
        Debug.Log($"[Audio Sync Calibrator] Offset set to: {offsetMs}ms");
    }
    
    /// <summary>
    /// Get current offset in seconds
    /// </summary>
    public float GetOffsetSeconds()
    {
        return offsetMs / 1000f;
    }
    
    /// <summary>
    /// Play test audio with current calibration
    /// </summary>
    public void PlayTestAudio()
    {
        if (testAudioClip == null || testA2FJson == null)
        {
            Debug.LogWarning("[Audio Sync Calibrator] Test audio or A2F data not set");
            return;
        }
        
        StartCoroutine(PlayTestSequence());
    }
    
    IEnumerator PlayTestSequence()
    {
        isCalibrating = true;
        
        // Load test data
        if (blendshapeDriver != null)
        {
            blendshapeDriver.LoadRuntimeJson(testA2FJson.text);
            blendshapeDriver.SetAudioClip(testAudioClip);
        }
        
        // Apply offset (positive = delay video, negative = delay audio)
        double dspStart = AudioSettings.dspTime + 0.1;
        
        if (offsetMs < 0)
        {
            // Delay audio
            dspStart += Mathf.Abs(offsetMs) / 1000.0;
            blendshapeDriver?.PlaySynced(0.1);
            yield return new WaitForSeconds(0.1f);
        }
        else if (offsetMs > 0)
        {
            // Delay video (start blendshapes later)
            audioSource.PlayScheduled(dspStart);
            yield return new WaitForSeconds((offsetMs / 1000f) + 0.1f);
            blendshapeDriver?.PlaySynced(0);
        }
        else
        {
            // No offset
            blendshapeDriver?.PlaySynced(0.1);
        }
        
        Debug.Log($"[Audio Sync Calibrator] Playing test with offset: {offsetMs}ms");
        
        // Wait for audio to finish
        yield return new WaitForSeconds(testAudioClip.length);
        
        isCalibrating = false;
        Debug.Log("[Audio Sync Calibrator] Test complete");
    }
    
    /// <summary>
    /// Stop test playback
    /// </summary>
    public void StopTest()
    {
        if (audioSource != null && audioSource.isPlaying)
        {
            audioSource.Stop();
        }
        
        if (blendshapeDriver != null)
        {
            blendshapeDriver.Stop();
        }
        
        isCalibrating = false;
        Debug.Log("[Audio Sync Calibrator] Test stopped");
    }
    
    /// <summary>
    /// Reset calibration to default
    /// </summary>
    public void ResetCalibration()
    {
        offsetMs = 0f;
        SaveCalibration();
        Debug.Log("[Audio Sync Calibrator] Calibration reset to 0ms");
    }
    
    /// <summary>
    /// Get calibration status
    /// </summary>
    public CalibrationStatus GetStatus()
    {
        return new CalibrationStatus
        {
            offsetMs = offsetMs,
            deviceId = GetDeviceIdentifier(),
            isCalibrated = Mathf.Abs(offsetMs) > 0.1f,
            isCalibrating = isCalibrating
        };
    }
    
    /// <summary>
    /// Auto-calibrate using audio analysis (experimental)
    /// </summary>
    public IEnumerator AutoCalibrate()
    {
        Debug.Log("[Audio Sync Calibrator] Starting auto-calibration...");
        
        float bestOffset = 0f;
        float bestScore = 0f;
        
        // Test offsets from -30ms to +30ms
        for (float testOffset = -30f; testOffset <= 30f; testOffset += 5f)
        {
            SetOffset(testOffset);
            PlayTestAudio();
            
            // Wait for test to complete
            yield return new WaitForSeconds(testAudioClip.length + 0.5f);
            
            // Calculate sync score (would need analysis implementation)
            float score = CalculateSyncScore();
            
            if (score > bestScore)
            {
                bestScore = score;
                bestOffset = testOffset;
            }
        }
        
        // Apply best offset
        SetOffset(bestOffset);
        SaveCalibration();
        
        Debug.Log($"[Audio Sync Calibrator] Auto-calibration complete. Best offset: {bestOffset}ms (score: {bestScore})");
    }
    
    /// <summary>
    /// Calculate lip-sync score (0-1, higher is better)
    /// Would need actual implementation with audio analysis
    /// </summary>
    float CalculateSyncScore()
    {
        // TODO: Implement actual sync scoring
        // Could use audio peak detection + visual motion analysis
        // For now, return placeholder
        return Random.Range(0.5f, 1f);
    }
    
    /// <summary>
    /// Get recommended offset for current device
    /// Based on known device profiles
    /// </summary>
    public float GetRecommendedOffset()
    {
        string model = SystemInfo.deviceModel.ToLower();
        
        // Known device offsets (to be populated from testing)
        if (model.Contains("iphone 12"))
            return -10f;
        else if (model.Contains("iphone 13"))
            return -8f;
        else if (model.Contains("iphone 14"))
            return -5f;
        else if (model.Contains("iphone 15"))
            return -3f;
        else if (model.Contains("pixel 6"))
            return 5f;
        else if (model.Contains("pixel 7"))
            return 3f;
        else if (model.Contains("pixel 8"))
            return 2f;
        
        // Default: no offset
        return 0f;
    }
    
    /// <summary>
    /// Apply recommended offset for current device
    /// </summary>
    public void ApplyRecommendedOffset()
    {
        float recommended = GetRecommendedOffset();
        SetOffset(recommended);
        SaveCalibration();
        Debug.Log($"[Audio Sync Calibrator] Applied recommended offset: {recommended}ms");
    }
}

/// <summary>
/// Calibration status information
/// </summary>
[System.Serializable]
public struct CalibrationStatus
{
    public float offsetMs;
    public string deviceId;
    public bool isCalibrated;
    public bool isCalibrating;
}


