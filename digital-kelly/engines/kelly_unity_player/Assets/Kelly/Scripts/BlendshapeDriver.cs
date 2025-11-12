using System;
using System.Collections.Generic;
using UnityEngine;

public class BlendshapeDriver : MonoBehaviour
{
    public SkinnedMeshRenderer headRenderer;
    public AudioSource audioSource;
    public TextAsset a2fJsonAsset;
    public float intensity = 100f;

    private Dictionary<string, int> shapeIndex = new();
    private A2FData data;
    private float frameTime;
    private double dspStart;
    private bool playing;
    private int blendShapeCount;

    void Awake()
    {
        if (headRenderer == null)
            headRenderer = GetComponentInChildren<SkinnedMeshRenderer>();
        
        if (a2fJsonAsset != null)
            LoadRuntimeJson(a2fJsonAsset.text);
        
        IndexBlendshapes();
    }

    void IndexBlendshapes()
    {
        if (headRenderer == null || headRenderer.sharedMesh == null)
            return;

        blendShapeCount = headRenderer.sharedMesh.blendShapeCount;
        
        for (int i = 0; i < blendShapeCount; i++)
        {
            var name = headRenderer.sharedMesh.GetBlendShapeName(i);
            var norm = Normalize(name);
            if (!shapeIndex.ContainsKey(norm))
                shapeIndex[norm] = i;
        }
    }

    string Normalize(string s) =>
        s.Replace(" ", "").Replace("_L", "_Left").Replace("_R", "_Right").ToLower();

    public void LoadRuntimeJson(string json)
    {
        data = JsonUtility.FromJson<A2FData>(json);
        int fps = Mathf.Max(1, data.fps);
        frameTime = 1f / fps;
    }

    public void SetAudioClip(AudioClip clip)
    {
        if (audioSource == null)
            audioSource = gameObject.GetComponent<AudioSource>() ?? gameObject.AddComponent<AudioSource>();

        audioSource.playOnAwake = false;
        audioSource.clip = clip;
    }

    public void PlaySynced(double startDelay = 0.05)
    {
        if (audioSource?.clip == null || data?.frames == null || data.frames.Count == 0)
            return;

        dspStart = AudioSettings.dspTime + startDelay;
        audioSource.PlayScheduled(dspStart);
        playing = true;
    }

    void Update()
    {
        if (!playing || data == null || data.frames == null)
            return;

        double t = Math.Max(0, AudioSettings.dspTime - dspStart);
        int frame = (int)(t / frameTime);
        
        if (frame >= data.frames.Count)
        {
            playing = false;
            return;
        }

        // Clear (optional)
        for (int i = 0; i < blendShapeCount; i++)
            headRenderer.SetBlendShapeWeight(i, 0f);

        var cur = data.frames[frame];

        foreach (var kv in cur)
        {
            var key = Normalize(kv.Key);
            
            if (shapeIndex.TryGetValue(key, out int idx))
            {
                headRenderer.SetBlendShapeWeight(idx, Mathf.Clamp01(kv.Value) * intensity);
            }
            else
            {
                // Simple aliases
                if (key == "blinkleft" && shapeIndex.TryGetValue("eyeblink_left", out idx))
                    headRenderer.SetBlendShapeWeight(idx, Mathf.Clamp01(kv.Value) * intensity);
                    
                if (key == "blinkright" && shapeIndex.TryGetValue("eyeblink_right", out idx))
                    headRenderer.SetBlendShapeWeight(idx, Mathf.Clamp01(kv.Value) * intensity);
            }
        }
    }
}


















