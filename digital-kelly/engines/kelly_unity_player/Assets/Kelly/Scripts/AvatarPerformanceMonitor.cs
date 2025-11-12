using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

/// <summary>
/// Monitors Kelly avatar performance - ensures 60fps target
/// </summary>
public class AvatarPerformanceMonitor : MonoBehaviour
{
    [Header("Performance Targets")]
    public float targetFPS = 60f;
    public float warningThreshold = 55f; // Warn if below this
    
    [Header("Monitoring")]
    public bool enableMonitoring = true;
    public bool logToConsole = false;
    public float sampleInterval = 1f;
    
    // Performance metrics
    private List<float> fpsHistory = new List<float>();
    private List<float> frameTimeHistory = new List<float>();
    private float lastSampleTime;
    
    // Memory tracking
    private long lastTotalMemory;
    private long lastUnityMemory;
    
    // Performance stats
    public struct PerformanceStats
    {
        public float currentFPS;
        public float averageFPS;
        public float minFPS;
        public float maxFPS;
        public float frameTimeMs;
        public long totalMemoryMB;
        public long unityMemoryMB;
        public bool meetingTarget;
        public string status;
    }
    
    private PerformanceStats currentStats;

    void Start()
    {
        lastSampleTime = Time.time;
        InvokeRepeating(nameof(SamplePerformance), sampleInterval, sampleInterval);
    }

    void Update()
    {
        if (!enableMonitoring)
            return;

        // Calculate FPS
        float currentFPS = 1f / Time.deltaTime;
        fpsHistory.Add(currentFPS);
        
        // Calculate frame time in ms
        float frameTimeMs = Time.deltaTime * 1000f;
        frameTimeHistory.Add(frameTimeMs);
        
        // Keep history limited (last 60 samples)
        if (fpsHistory.Count > 60)
        {
            fpsHistory.RemoveAt(0);
            frameTimeHistory.RemoveAt(0);
        }
    }

    void SamplePerformance()
    {
        if (!enableMonitoring || fpsHistory.Count == 0)
            return;

        // Calculate stats
        float avgFPS = 0f;
        float minFPS = float.MaxValue;
        float maxFPS = float.MinValue;
        
        foreach (float fps in fpsHistory)
        {
            avgFPS += fps;
            if (fps < minFPS) minFPS = fps;
            if (fps > maxFPS) maxFPS = fps;
        }
        
        avgFPS /= fpsHistory.Count;
        
        float avgFrameTime = 0f;
        foreach (float ft in frameTimeHistory)
            avgFrameTime += ft;
        avgFrameTime /= frameTimeHistory.Count;
        
        // Memory stats
        long totalMemory = System.GC.GetTotalMemory(false) / (1024 * 1024);
        long unityMemory = Profiler.GetTotalAllocatedMemoryLong() / (1024 * 1024);
        
        // Determine status
        bool meetingTarget = avgFPS >= targetFPS * 0.95f; // Allow 5% variance
        string status;
        
        if (avgFPS >= targetFPS)
            status = "Excellent";
        else if (avgFPS >= warningThreshold)
            status = "Good";
        else if (avgFPS >= targetFPS * 0.75f)
            status = "Warning";
        else
            status = "Poor";
        
        // Update current stats
        currentStats = new PerformanceStats
        {
            currentFPS = 1f / Time.deltaTime,
            averageFPS = avgFPS,
            minFPS = minFPS,
            maxFPS = maxFPS,
            frameTimeMs = avgFrameTime,
            totalMemoryMB = totalMemory,
            unityMemoryMB = unityMemory,
            meetingTarget = meetingTarget,
            status = status
        };
        
        // Log if enabled
        if (logToConsole)
        {
            Debug.Log($"[Performance] FPS: {avgFPS:F1} (min: {minFPS:F1}, max: {maxFPS:F1}), " +
                     $"Frame Time: {avgFrameTime:F2}ms, " +
                     $"Memory: {unityMemory}MB, " +
                     $"Status: {status}");
        }
        
        // Warning if below threshold
        if (!meetingTarget && logToConsole)
        {
            Debug.LogWarning($"[Performance] Below 60fps target! Current: {avgFPS:F1}fps");
        }
    }

    public PerformanceStats GetCurrentStats()
    {
        return currentStats;
    }

    public bool IsMeetingTarget()
    {
        return currentStats.meetingTarget;
    }

    public float GetAverageFPS()
    {
        return currentStats.averageFPS;
    }

    void OnGUI()
    {
        if (!enableMonitoring)
            return;

        // Performance overlay
        GUILayout.BeginArea(new Rect(Screen.width - 320, 10, 310, 180));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label("Kelly Performance Monitor", GUI.skin.GetStyle("boldLabel"));
        GUILayout.Space(5);
        
        // FPS display with color coding
        GUI.color = currentStats.meetingTarget ? Color.green : Color.yellow;
        GUILayout.Label($"FPS: {currentStats.currentFPS:F1} (avg: {currentStats.averageFPS:F1})");
        GUI.color = Color.white;
        
        GUILayout.Label($"Min/Max: {currentStats.minFPS:F1} / {currentStats.maxFPS:F1}");
        GUILayout.Label($"Frame Time: {currentStats.frameTimeMs:F2}ms");
        GUILayout.Label($"Target: 16.67ms (60fps)");
        GUILayout.Label($"Memory: {currentStats.unityMemoryMB}MB");
        
        GUILayout.Space(5);
        
        // Status indicator
        string statusColor = currentStats.status switch
        {
            "Excellent" => "#00FF00",
            "Good" => "#FFFF00",
            "Warning" => "#FFA500",
            _ => "#FF0000"
        };
        
        GUILayout.Label($"<color={statusColor}>Status: {currentStats.status}</color>", 
                       new GUIStyle(GUI.skin.label) { richText = true });
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }

    /// <summary>
    /// Get detailed performance report
    /// </summary>
    public string GetPerformanceReport()
    {
        return $@"Kelly Avatar Performance Report
================================
Current FPS: {currentStats.currentFPS:F1}
Average FPS: {currentStats.averageFPS:F1}
Min FPS: {currentStats.minFPS:F1}
Max FPS: {currentStats.maxFPS:F1}
Frame Time: {currentStats.frameTimeMs:F2}ms (target: 16.67ms)
Memory Usage: {currentStats.unityMemoryMB}MB
Target Met: {(currentStats.meetingTarget ? "YES" : "NO")}
Status: {currentStats.status}
================================";
    }
}














