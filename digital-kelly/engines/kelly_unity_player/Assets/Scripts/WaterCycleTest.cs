using UnityEngine;

/// <summary>
/// Quick test script to play water-cycle lesson audio through Kelly
/// </summary>
public class WaterCycleTest : MonoBehaviour
{
    [Header("Test Settings")]
    public string lessonId = "water-cycle";
    public string ageGroup = "6-12"; // Kid-friendly
    public string language = "en";
    public bool playOnStart = true;
    
    private LessonAudioPlayer audioPlayer;
    private AudioSource audioSource;
    
    void Start()
    {
        Debug.Log("=== Water Cycle Test Starting ===");
        
        // Find or create audio player
        audioPlayer = FindObjectOfType<LessonAudioPlayer>();
        if (audioPlayer == null)
        {
            GameObject playerObj = new GameObject("LessonAudioPlayer");
            audioPlayer = playerObj.AddComponent<LessonAudioPlayer>();
            audioSource = playerObj.AddComponent<AudioSource>();
            audioPlayer.audioSource = audioSource;
        }
        
        // Load the water-cycle audio
        Debug.Log($"Loading lesson: {lessonId} for age {ageGroup} in {language}");
        audioPlayer.LoadLessonAudio(lessonId, $"{ageGroup}-welcome-{language}");
        
        // Try to load audio clips manually
        LoadAudioClipsManually();
        
        // Play if enabled
        if (playOnStart)
        {
            Debug.Log("Auto-playing welcome section in 2 seconds...");
            Invoke("PlayWelcome", 2f);
        }
    }
    
    void LoadAudioClipsManually()
    {
        // Try different path formats to find the audio
        string[] pathsToTry = new string[]
        {
            $"Audio/Lessons/{lessonId}/{ageGroup}-welcome-{language}",
            $"Audio/Lessons/{lessonId}/{ageGroup}-welcome",
            $"Audio/Lessons/water-cycle/6-12-welcome-en",
            "Audio/Lessons/water-cycle/6-12-welcome",
        };
        
        Debug.Log("Attempting to load audio from Resources...");
        foreach (string path in pathsToTry)
        {
            AudioClip clip = Resources.Load<AudioClip>(path);
            if (clip != null)
            {
                Debug.Log($"✓ Found audio at: {path}");
                audioPlayer.welcomeClip = clip;
                return;
            }
            else
            {
                Debug.Log($"✗ Not found: {path}");
            }
        }
        
        Debug.LogWarning("Could not find audio clips in Resources folder");
    }
    
    void PlayWelcome()
    {
        if (audioPlayer != null && audioPlayer.welcomeClip != null)
        {
            Debug.Log("Playing welcome section!");
            audioPlayer.PlaySection(LessonSection.Welcome);
        }
        else
        {
            Debug.LogError("Cannot play - audio player or clip not ready");
            if (audioPlayer == null) Debug.LogError("  - audioPlayer is null");
            else if (audioPlayer.welcomeClip == null) Debug.LogError("  - welcomeClip is null");
        }
    }
    
    // Manual test buttons
    [ContextMenu("Test: Play Welcome")]
    public void TestPlayWelcome()
    {
        PlayWelcome();
    }
    
    [ContextMenu("Test: Stop Audio")]
    public void TestStopAudio()
    {
        if (audioPlayer != null)
        {
            audioPlayer.Stop();
        }
    }
    
    [ContextMenu("Test: List Resources")]
    public void TestListResources()
    {
        Debug.Log("=== Checking Resources Folder ===");
        Object[] audioClips = Resources.LoadAll("Audio/Lessons/water-cycle", typeof(AudioClip));
        Debug.Log($"Found {audioClips.Length} audio clips in water-cycle folder");
        foreach (Object obj in audioClips)
        {
            Debug.Log($"  - {obj.name}");
        }
    }
}

