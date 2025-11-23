using UnityEngine;

/// <summary>
/// Simple audio test for Kelly - just plays audio without lip-sync for now
/// </summary>
public class SimpleKellyAudioTest : MonoBehaviour
{
    [Header("Audio Settings")]
    public AudioClip audioClip;
    public float playDelay = 2f;
    
    private AudioSource audioSource;
    private SkinnedMeshRenderer headRenderer;
    
    void Start()
    {
        Debug.Log("=== Simple Kelly Audio Test ===");
        
        // Find Kelly's head renderer (for future lip-sync)
        headRenderer = GetComponentInChildren<SkinnedMeshRenderer>();
        if (headRenderer != null)
        {
            Debug.Log($"✓ Found head renderer with {headRenderer.sharedMesh.blendShapeCount} blendshapes");
            LogBlendshapes();
        }
        else
        {
            Debug.LogWarning("⚠ No SkinnedMeshRenderer found");
        }
        
        // Setup audio source
        audioSource = gameObject.GetComponent<AudioSource>();
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
        }
        
        audioSource.playOnAwake = false;
        audioSource.clip = audioClip;
        
        if (audioClip != null)
        {
            Debug.Log($"✓ Audio clip loaded: {audioClip.name} ({audioClip.length:F1}s)");
            Invoke("PlayAudio", playDelay);
        }
        else
        {
            Debug.LogError("✗ No audio clip assigned!");
        }
    }
    
    void PlayAudio()
    {
        if (audioSource != null && audioSource.clip != null)
        {
            audioSource.Play();
            Debug.Log("▶ Playing audio!");
        }
    }
    
    void LogBlendshapes()
    {
        if (headRenderer == null || headRenderer.sharedMesh == null) return;
        
        int count = headRenderer.sharedMesh.blendShapeCount;
        Debug.Log($"=== Kelly Blendshapes ({count} total) ===");
        
        // Log first 10 for reference
        int logCount = Mathf.Min(10, count);
        for (int i = 0; i < logCount; i++)
        {
            string name = headRenderer.sharedMesh.GetBlendShapeName(i);
            Debug.Log($"  [{i}] {name}");
        }
        
        if (count > 10)
        {
            Debug.Log($"  ... and {count - 10} more");
        }
    }
    
    [ContextMenu("Play Audio Now")]
    public void PlayNow()
    {
        if (audioSource != null && audioSource.clip != null)
        {
            audioSource.Stop();
            audioSource.Play();
        }
    }
    
    [ContextMenu("Stop Audio")]
    public void Stop()
    {
        if (audioSource != null)
        {
            audioSource.Stop();
        }
    }
}




