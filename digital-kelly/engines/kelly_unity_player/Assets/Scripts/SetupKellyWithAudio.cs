using UnityEngine;

/// <summary>
/// One-click setup: Add Kelly to scene with lip-sync audio playback
/// </summary>
public class SetupKellyWithAudio : MonoBehaviour
{
    [Header("Kelly Model")]
    public GameObject kellyPrefab;
    public Transform kellySpawnPosition;
    
    [Header("Audio Test")]
    public AudioClip testAudioClip;
    public TextAsset a2fJsonData; // Optional: A2F lip-sync data
    
    [Header("Auto Setup")]
    public bool setupOnStart = true;
    
    private GameObject kellyInstance;
    private BlendshapeDriver blendshapeDriver;
    private AudioSource audioSource;
    
    void Start()
    {
        if (setupOnStart)
        {
            SetupKelly();
        }
    }
    
    [ContextMenu("Setup Kelly in Scene")]
    public void SetupKelly()
    {
        Debug.Log("=== Setting up Kelly ===");
        
        // 1. Check if Kelly prefab is assigned
        if (kellyPrefab == null)
        {
            Debug.LogError("Kelly prefab not assigned! Please drag the Kelly model from Assets/Kelly/Models into the Kelly Prefab field in the Inspector.");
            return;
        }
        
        // 2. Instantiate Kelly
        Vector3 spawnPos = kellySpawnPosition != null ? kellySpawnPosition.position : new Vector3(0, 0, 0);
        kellyInstance = Instantiate(kellyPrefab, spawnPos, Quaternion.identity);
        kellyInstance.name = "Kelly_Avatar";
        
        Debug.Log($"✓ Kelly spawned at {spawnPos}");
        
        // 3. Find or add BlendshapeDriver
        blendshapeDriver = kellyInstance.GetComponent<BlendshapeDriver>();
        if (blendshapeDriver == null)
        {
            blendshapeDriver = kellyInstance.AddComponent<BlendshapeDriver>();
            Debug.Log("✓ Added BlendshapeDriver");
        }
        
        // 4. Find Kelly's head renderer
        SkinnedMeshRenderer headRenderer = kellyInstance.GetComponentInChildren<SkinnedMeshRenderer>();
        if (headRenderer != null)
        {
            blendshapeDriver.headRenderer = headRenderer;
            Debug.Log($"✓ Found head renderer with {headRenderer.sharedMesh.blendShapeCount} blendshapes");
        }
        else
        {
            Debug.LogWarning("⚠ Could not find SkinnedMeshRenderer on Kelly");
        }
        
        // 5. Setup audio source
        audioSource = kellyInstance.GetComponent<AudioSource>();
        if (audioSource == null)
        {
            audioSource = kellyInstance.AddComponent<AudioSource>();
        }
        audioSource.playOnAwake = false;
        blendshapeDriver.audioSource = audioSource;
        
        Debug.Log("✓ Audio source configured");
        
        // 6. Assign A2F data if available
        if (a2fJsonData != null)
        {
            blendshapeDriver.a2fJsonAsset = a2fJsonData;
            blendshapeDriver.LoadRuntimeJson(a2fJsonData.text);
            Debug.Log("✓ Loaded A2F lip-sync data");
        }
        else
        {
            Debug.LogWarning("⚠ No A2F data assigned - lip-sync will not work");
        }
        
        // 7. Play test audio if assigned
        if (testAudioClip != null)
        {
            Debug.Log($"Playing test audio: {testAudioClip.name}");
            blendshapeDriver.SetAudioClip(testAudioClip);
            Invoke("PlayAudio", 2f);
        }
        
        Debug.Log("=== Kelly setup complete! ===");
    }
    
    void PlayAudio()
    {
        if (blendshapeDriver != null)
        {
            blendshapeDriver.PlaySynced();
            Debug.Log("▶ Playing audio with lip-sync");
        }
    }
    
    [ContextMenu("Play Audio")]
    public void ManualPlayAudio()
    {
        if (blendshapeDriver != null && testAudioClip != null)
        {
            blendshapeDriver.SetAudioClip(testAudioClip);
            blendshapeDriver.PlaySynced();
        }
    }
    
    [ContextMenu("Stop Audio")]
    public void StopAudio()
    {
        if (blendshapeDriver != null)
        {
            blendshapeDriver.Stop();
        }
    }
}

