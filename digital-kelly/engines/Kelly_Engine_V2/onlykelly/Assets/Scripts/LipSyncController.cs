using UnityEngine;
using System.Collections.Generic;
using System.Linq;

[RequireComponent(typeof(AudioSource))]
public class LipSyncController : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private ARKitBlendshapeController blendshapes;
    [SerializeField] private AudioSource audioSource;
    
    [Header("Phoneme Mapping")]
    private Dictionary<string, string[]> phonemeToBlendshapes = new Dictionary<string, string[]>()
    {
        // Viseme → ARKit blendshapes mapping
        {"A", new[] {"jawOpen", "mouthOpen"}},
        {"E", new[] {"mouthSmile_L", "mouthSmile_R"}},
        {"I", new[] {"mouthSmile_L", "mouthSmile_R", "mouthClose"}},
        {"O", new[] {"mouthFunnel", "jawOpen"}},
        {"U", new[] {"mouthPucker", "mouthFunnel"}},
        {"F", new[] {"mouthLowerDown_L", "mouthLowerDown_R"}},
        {"M", new[] {"mouthClose", "mouthPress_L", "mouthPress_R"}},
        {"S", new[] {"mouthSmile_L", "mouthSmile_R", "jawOpen"}},
    };
    
    [Header("Settings")]
    [SerializeField] [Range(0f, 1f)] private float lipSyncStrength = 1f;
    [SerializeField] [Range(0f, 1f)] private float smoothing = 0.3f;
    
    private float[] spectrumData = new float[256];
    private Dictionary<string, float> currentWeights = new Dictionary<string, float>();
    
    public void StartLipSync(AudioClip clip)
    {
        // Reset
        ResetLipSync();
        
        // Start analyzing
        enabled = true;
    }
    
    private void Update()
    {
        if (!audioSource.isPlaying)
        {
            ResetLipSync();
            enabled = false;
            return;
        }
        
        // Get spectrum data
        audioSource.GetSpectrumData(spectrumData, 0, FFTWindow.BlackmanHarris);
        
        // Analyze frequencies to determine phoneme
        string phoneme = AnalyzePhoneme(spectrumData);
        
        // Apply blendshapes
        ApplyPhonemeBlendshapes(phoneme);
    }
    
    private string AnalyzePhoneme(float[] spectrum)
    {
        // Simple frequency analysis (you can use more sophisticated methods)
        float low = 0f, mid = 0f, high = 0f;
        
        // Low frequencies (vowels)
        for (int i = 0; i < 20; i++)
            low += spectrum[i];
        
        // Mid frequencies
        for (int i = 20; i < 80; i++)
            mid += spectrum[i];
        
        // High frequencies (consonants)
        for (int i = 80; i < 256; i++)
            high += spectrum[i];
        
        // Determine phoneme based on frequency distribution
        if (low > mid && low > high)
        {
            if (mid > high) return "O"; // Round vowel
            else return "A"; // Open vowel
        }
        else if (mid > low && mid > high)
        {
            return "E"; // Mid vowel
        }
        else if (high > low && high > mid)
        {
            if (mid > low) return "S"; // Sibilant
            else return "F"; // Fricative
        }
        else if (low + mid + high < 0.1f)
        {
            return "M"; // Closed mouth (silence)
        }
        
        return ""; // Neutral
    }
    
    private void ApplyPhonemeBlendshapes(string phoneme)
    {
        if (string.IsNullOrEmpty(phoneme)) return;
        if (!phonemeToBlendshapes.ContainsKey(phoneme)) return;
        
        string[] targetBlendshapes = phonemeToBlendshapes[phoneme];
        
        // Smooth transition
        foreach (var kvp in currentWeights.ToArray())
        {
            float targetWeight = System.Array.IndexOf(targetBlendshapes, kvp.Key) >= 0 
                ? lipSyncStrength * 100f 
                : 0f;
            
            float newWeight = Mathf.Lerp(kvp.Value, targetWeight, smoothing);
            currentWeights[kvp.Key] = newWeight;
            blendshapes.SetBlendshape(kvp.Key, newWeight);
        }
        
        // Add new blendshapes
        foreach (string bs in targetBlendshapes)
        {
            if (!currentWeights.ContainsKey(bs))
            {
                currentWeights[bs] = lipSyncStrength * 100f;
                blendshapes.SetBlendshape(bs, currentWeights[bs]);
            }
        }
    }
    
    private void ResetLipSync()
    {
        foreach (var kvp in currentWeights)
        {
            blendshapes.SetBlendshape(kvp.Key, 0f);
        }
        currentWeights.Clear();
    }
}

