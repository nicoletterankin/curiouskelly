using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Simple test script to get Kelly talking in Unity
/// Attach this to the Kelly avatar GameObject for quick testing
/// </summary>
public class KellyTalkTest : MonoBehaviour
{
    [Header("Components")]
    public KellyAvatarController avatarController;
    public BlendshapeDriver60fps blendshapeDriver;
    public LessonAudioPlayer audioPlayer;
    
    [Header("Test Audio")]
    public AudioClip testAudioClip; // Drag an audio clip here for testing
    
    [Header("Test Settings")]
    public bool autoPlayOnStart = false;
    public string testText = "Hello! I'm Kelly, and I'm here to help you learn.";
    
    private AudioSource audioSource;
    
    void Start()
    {
        // Auto-find components if not assigned
        if (avatarController == null)
            avatarController = GetComponent<KellyAvatarController>();
        
        if (blendshapeDriver == null)
            blendshapeDriver = GetComponent<BlendshapeDriver60fps>();
        
        if (audioPlayer == null)
            audioPlayer = GetComponent<LessonAudioPlayer>();
        
        // Get or create AudioSource
        audioSource = GetComponent<AudioSource>();
        if (audioSource == null)
            audioSource = gameObject.AddComponent<AudioSource>();
        
        // Set up audio source
        audioSource.playOnAwake = false;
        audioSource.spatialBlend = 0f; // 2D sound
        
        // Assign to blendshape driver if needed
        if (blendshapeDriver != null && blendshapeDriver.audioSource == null)
        {
            blendshapeDriver.audioSource = audioSource;
        }
        
        Debug.Log("[KellyTalkTest] Setup complete. Press SPACE to test audio playback.");
        
        if (autoPlayOnStart && testAudioClip != null)
        {
            PlayTestAudio();
        }
    }
    
    void Update()
    {
        // Press SPACE to play test audio (works with both Input systems)
        bool spacePressed = false;
        bool tPressed = false;
        bool pPressed = false;
        bool sPressed = false;
        
        #if ENABLE_INPUT_SYSTEM
        // New Input System
        var keyboard = UnityEngine.InputSystem.Keyboard.current;
        if (keyboard != null)
        {
            spacePressed = keyboard.spaceKey.wasPressedThisFrame;
            tPressed = keyboard.tKey.wasPressedThisFrame;
            pPressed = keyboard.pKey.wasPressedThisFrame;
            sPressed = keyboard.sKey.wasPressedThisFrame;
        }
        #else
        // Old Input System
        spacePressed = Input.GetKeyDown(KeyCode.Space);
        tPressed = Input.GetKeyDown(KeyCode.T);
        pPressed = Input.GetKeyDown(KeyCode.P);
        sPressed = Input.GetKeyDown(KeyCode.S);
        #endif
        
        if (spacePressed)
        {
            if (testAudioClip != null)
            {
                PlayTestAudio();
            }
            else
            {
                Debug.LogWarning("[KellyTalkTest] No test audio clip assigned! Drag an audio clip to the Test Audio Clip field.");
            }
        }
        
        // Press T to trigger test speech
        if (tPressed)
        {
            if (avatarController != null)
            {
                avatarController.Speak(testText, 35);
            }
        }
        
        // Press P to pause/resume
        if (pPressed)
        {
            if (audioSource.isPlaying)
            {
                audioSource.Pause();
                Debug.Log("[KellyTalkTest] Paused");
            }
            else if (audioSource.clip != null)
            {
                audioSource.UnPause();
                Debug.Log("[KellyTalkTest] Resumed");
            }
        }
        
        // Press S to stop
        if (sPressed)
        {
            audioSource.Stop();
            Debug.Log("[KellyTalkTest] Stopped");
        }
    }
    
    public void PlayTestAudio()
    {
        if (testAudioClip == null)
        {
            Debug.LogError("[KellyTalkTest] No test audio clip assigned!");
            return;
        }
        
        Debug.Log($"[KellyTalkTest] Playing audio: {testAudioClip.name}");
        
        audioSource.clip = testAudioClip;
        audioSource.Play();
        
        // If blendshape driver has A2F data, sync it
        if (blendshapeDriver != null && blendshapeDriver.a2fJsonAsset != null)
        {
            blendshapeDriver.SetAudioClip(testAudioClip);
            blendshapeDriver.PlaySynced();
        }
        
        Debug.Log("[KellyTalkTest] Audio playing! Kelly should be talking now.");
    }
    
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label("Kelly Talk Test", GUI.skin.label);
        GUILayout.Space(5);
        
        GUILayout.Label("Controls:");
        GUILayout.Label("SPACE - Play test audio");
        GUILayout.Label("T - Trigger test speech");
        GUILayout.Label("P - Pause/Resume");
        GUILayout.Label("S - Stop");
        
        GUILayout.Space(10);
        
        if (audioSource != null)
        {
            GUILayout.Label($"Playing: {(audioSource.isPlaying ? "Yes" : "No")}");
            if (audioSource.clip != null)
            {
                GUILayout.Label($"Clip: {audioSource.clip.name}");
                GUILayout.Label($"Time: {audioSource.time:F1}s / {audioSource.clip.length:F1}s");
            }
        }
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
}
