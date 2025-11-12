using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Plays lesson audio files and syncs with Kelly avatar
/// </summary>
public class LessonAudioPlayer : MonoBehaviour
{
    [Header("Audio Settings")]
    public AudioSource audioSource;
    public bool autoPlayOnLoad = false;
    
    [Header("Current Lesson")]
    public string lessonId = "water-cycle";
    public string ageGroup = "18-35";
    
    [Header("Audio Clips")]
    public AudioClip welcomeClip;
    public AudioClip mainContentClip;
    public AudioClip wisdomMomentClip;
    
    [Header("Playback State")]
    public bool isPlaying = false;
    public LessonSection currentSection = LessonSection.None;
    public float progress = 0f;
    
    // Callbacks
    public Action<LessonSection> OnSectionStarted;
    public Action<LessonSection> OnSectionCompleted;
    public Action OnLessonCompleted;
    public Action<float> OnProgressUpdate;
    
    // Private state
    private Queue<LessonSection> playQueue = new Queue<LessonSection>();
    private float sectionStartTime = 0f;
    
    void Awake()
    {
        if (audioSource == null)
        {
            audioSource = gameObject.GetComponent<AudioSource>() ?? gameObject.AddComponent<AudioSource>();
        }
        
        audioSource.playOnAwake = false;
    }
    
    void Start()
    {
        if (autoPlayOnLoad)
        {
            PlayCompleteLesson();
        }
    }
    
    void Update()
    {
        if (isPlaying && audioSource.isPlaying)
        {
            // Update progress
            progress = audioSource.time / audioSource.clip.length;
            OnProgressUpdate?.Invoke(progress);
            
            // Check if section completed
            if (!audioSource.isPlaying && currentSection != LessonSection.None)
            {
                CompleteSection();
            }
        }
    }
    
    /// <summary>
    /// Play complete lesson (all sections in order)
    /// </summary>
    public void PlayCompleteLesson()
    {
        Debug.Log($"[LessonAudioPlayer] Playing complete lesson: {lessonId}");
        
        playQueue.Clear();
        playQueue.Enqueue(LessonSection.Welcome);
        playQueue.Enqueue(LessonSection.MainContent);
        playQueue.Enqueue(LessonSection.WisdomMoment);
        
        PlayNextInQueue();
    }
    
    /// <summary>
    /// Play specific section
    /// </summary>
    public void PlaySection(LessonSection section)
    {
        Debug.Log($"[LessonAudioPlayer] Playing section: {section}");
        
        AudioClip clip = GetClipForSection(section);
        
        if (clip == null)
        {
            Debug.LogError($"[LessonAudioPlayer] No audio clip for section: {section}");
            return;
        }
        
        currentSection = section;
        isPlaying = true;
        sectionStartTime = Time.time;
        
        audioSource.clip = clip;
        audioSource.Play();
        
        OnSectionStarted?.Invoke(section);
        
        Debug.Log($"[LessonAudioPlayer] Started: {section} ({clip.length}s)");
    }
    
    private void PlayNextInQueue()
    {
        if (playQueue.Count > 0)
        {
            LessonSection nextSection = playQueue.Dequeue();
            PlaySection(nextSection);
        }
        else
        {
            CompleteLesson();
        }
    }
    
    private void CompleteSection()
    {
        Debug.Log($"[LessonAudioPlayer] Completed section: {currentSection}");
        
        OnSectionCompleted?.Invoke(currentSection);
        
        currentSection = LessonSection.None;
        isPlaying = false;
        progress = 0f;
        
        // Play next in queue
        if (playQueue.Count > 0)
        {
            PlayNextInQueue();
        }
        else
        {
            CompleteLesson();
        }
    }
    
    private void CompleteLesson()
    {
        Debug.Log($"[LessonAudioPlayer] Lesson completed!");
        OnLessonCompleted?.Invoke();
    }
    
    /// <summary>
    /// Pause playback
    /// </summary>
    public void Pause()
    {
        if (audioSource.isPlaying)
        {
            audioSource.Pause();
            Debug.Log("[LessonAudioPlayer] Paused");
        }
    }
    
    /// <summary>
    /// Resume playback
    /// </summary>
    public void Resume()
    {
        if (!audioSource.isPlaying && audioSource.clip != null)
        {
            audioSource.UnPause();
            Debug.Log("[LessonAudioPlayer] Resumed");
        }
    }
    
    /// <summary>
    /// Stop playback
    /// </summary>
    public void Stop()
    {
        audioSource.Stop();
        playQueue.Clear();
        currentSection = LessonSection.None;
        isPlaying = false;
        progress = 0f;
        Debug.Log("[LessonAudioPlayer] Stopped");
    }
    
    /// <summary>
    /// Set volume (0-1)
    /// </summary>
    public void SetVolume(float volume)
    {
        audioSource.volume = Mathf.Clamp01(volume);
    }
    
    /// <summary>
    /// Load audio clips for specific lesson and age
    /// </summary>
    public void LoadLessonAudio(string newLessonId, string newAgeGroup)
    {
        lessonId = newLessonId;
        ageGroup = newAgeGroup;
        
        // Load from Resources folder
        // Path format: Audio/Lessons/{lessonId}/{ageGroup}-{section}
        string basePath = $"Audio/Lessons/{lessonId}";
        
        welcomeClip = Resources.Load<AudioClip>($"{basePath}/{ageGroup}-welcome");
        mainContentClip = Resources.Load<AudioClip>($"{basePath}/{ageGroup}-mainContent");
        wisdomMomentClip = Resources.Load<AudioClip>($"{basePath}/{ageGroup}-wisdomMoment");
        
        Debug.Log($"[LessonAudioPlayer] Loaded audio for {lessonId} / {ageGroup}");
        Debug.Log($"  Welcome: {(welcomeClip != null ? "✓" : "✗")}");
        Debug.Log($"  MainContent: {(mainContentClip != null ? "✓" : "✗")}");
        Debug.Log($"  WisdomMoment: {(wisdomMomentClip != null ? "✓" : "✗")}");
    }
    
    private AudioClip GetClipForSection(LessonSection section)
    {
        switch (section)
        {
            case LessonSection.Welcome:
                return welcomeClip;
            case LessonSection.MainContent:
                return mainContentClip;
            case LessonSection.WisdomMoment:
                return wisdomMomentClip;
            default:
                return null;
        }
    }
    
    /// <summary>
    /// Get current playback time
    /// </summary>
    public float GetCurrentTime()
    {
        return audioSource.time;
    }
    
    /// <summary>
    /// Get total duration of current section
    /// </summary>
    public float GetDuration()
    {
        return audioSource.clip != null ? audioSource.clip.length : 0f;
    }
    
    /// <summary>
    /// Get time remaining in current section
    /// </summary>
    public float GetTimeRemaining()
    {
        return GetDuration() - GetCurrentTime();
    }
}

public enum LessonSection
{
    None,
    Welcome,
    MainContent,
    WisdomMoment
}





