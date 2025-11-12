using UnityEngine;
using TMPro;

/// <summary>
/// FPS Counter for performance monitoring
/// Displays current, average, min, and max frame rates
/// </summary>
public class FPSCounter : MonoBehaviour
{
    [Header("Display Settings")]
    public bool showDebugUI = true;
    public KeyCode toggleKey = KeyCode.F3;
    
    [Header("Update Settings")]
    public float updateInterval = 0.5f; // Update display every 0.5s
    
    // FPS tracking
    private float deltaTime = 0f;
    private float fps = 0f;
    private float avgFps = 0f;
    private float minFps = float.MaxValue;
    private float maxFps = 0f;
    
    // Timing
    private float timeSinceLastUpdate = 0f;
    private int frames = 0;
    
    // Performance warnings
    private const float TARGET_FPS = 60f;
    private const float WARNING_THRESHOLD = 50f;
    private bool isPerformanceWarning = false;
    
    // GUI Style
    private GUIStyle style;
    private Rect rect;
    private string displayText = "";
    
    void Start()
    {
        // Initialize GUI style
        style = new GUIStyle();
        style.alignment = TextAnchor.UpperLeft;
        style.fontSize = 24;
        style.normal.textColor = Color.green;
        
        rect = new Rect(10, 10, 400, 150);
        
        // Set target frame rate
        Application.targetFrameRate = (int)TARGET_FPS;
        QualitySettings.vSyncCount = 1; // Enable VSync for mobile
        
        Debug.Log($"[FPS Counter] Target frame rate set to {TARGET_FPS} FPS");
    }
    
    void Update()
    {
        // Toggle display
        if (Input.GetKeyDown(toggleKey))
        {
            showDebugUI = !showDebugUI;
        }
        
        // Calculate delta time
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        
        // Count frames
        frames++;
        timeSinceLastUpdate += Time.unscaledDeltaTime;
        
        // Update display at interval
        if (timeSinceLastUpdate >= updateInterval)
        {
            // Calculate FPS
            fps = frames / timeSinceLastUpdate;
            
            // Update statistics
            if (fps < minFps) minFps = fps;
            if (fps > maxFps) maxFps = fps;
            avgFps = (avgFps == 0) ? fps : (avgFps * 0.9f + fps * 0.1f);
            
            // Check performance warning
            isPerformanceWarning = fps < WARNING_THRESHOLD;
            
            // Update display text
            UpdateDisplayText();
            
            // Reset counters
            frames = 0;
            timeSinceLastUpdate = 0f;
        }
    }
    
    void UpdateDisplayText()
    {
        displayText = $"FPS: {fps:F1}\n";
        displayText += $"Avg: {avgFps:F1}\n";
        displayText += $"Min: {minFps:F1}\n";
        displayText += $"Max: {maxFps:F1}\n";
        displayText += $"MS: {deltaTime * 1000f:F1}ms";
        
        if (isPerformanceWarning)
        {
            displayText += "\n⚠️ LOW FPS";
        }
        
        // Update color
        if (fps >= TARGET_FPS * 0.95f)
        {
            style.normal.textColor = Color.green; // Good
        }
        else if (fps >= WARNING_THRESHOLD)
        {
            style.normal.textColor = Color.yellow; // Acceptable
        }
        else
        {
            style.normal.textColor = Color.red; // Poor
        }
    }
    
    void OnGUI()
    {
        if (!showDebugUI) return;
        
        // Draw semi-transparent background
        GUI.Box(rect, "");
        
        // Draw FPS text
        GUI.Label(rect, displayText, style);
    }
    
    /// <summary>
    /// Get current FPS
    /// </summary>
    public float GetCurrentFPS()
    {
        return fps;
    }
    
    /// <summary>
    /// Get average FPS
    /// </summary>
    public float GetAverageFPS()
    {
        return avgFps;
    }
    
    /// <summary>
    /// Check if performance is good (>= target)
    /// </summary>
    public bool IsPerformanceGood()
    {
        return fps >= TARGET_FPS * 0.95f;
    }
    
    /// <summary>
    /// Reset statistics
    /// </summary>
    public void ResetStats()
    {
        minFps = float.MaxValue;
        maxFps = 0f;
        avgFps = 0f;
        Debug.Log("[FPS Counter] Statistics reset");
    }
    
    /// <summary>
    /// Log performance report
    /// </summary>
    public void LogPerformanceReport()
    {
        Debug.Log("=== PERFORMANCE REPORT ===");
        Debug.Log($"Current FPS: {fps:F1}");
        Debug.Log($"Average FPS: {avgFps:F1}");
        Debug.Log($"Min FPS: {minFps:F1}");
        Debug.Log($"Max FPS: {maxFps:F1}");
        Debug.Log($"Frame Time: {deltaTime * 1000f:F1}ms");
        Debug.Log($"Target Met: {(IsPerformanceGood() ? "YES" : "NO")}");
        Debug.Log("========================");
    }
}


