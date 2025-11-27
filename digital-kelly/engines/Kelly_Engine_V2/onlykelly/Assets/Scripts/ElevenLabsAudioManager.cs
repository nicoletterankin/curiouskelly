using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;

public class ElevenLabsAudioManager : MonoBehaviour
{
    [Header("ElevenLabs Configuration")]
    [SerializeField] private string apiKey = "YOUR_API_KEY";
    [SerializeField] private string voiceId = "YOUR_VOICE_ID";
    
    [Header("Audio Components")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private ARKitBlendshapeController blendshapeController;
    
    [Header("Lip Sync")]
    [SerializeField] private LipSyncController lipSync;
    
    private const string API_URL = "https://api.elevenlabs.io/v1/text-to-speech/";
    
    public void SpeakText(string text, System.Action onComplete = null)
    {
        StartCoroutine(GenerateAndPlayAudio(text, onComplete));
    }
    
    private IEnumerator GenerateAndPlayAudio(string text, System.Action onComplete)
    {
        // 1. Request audio from ElevenLabs
        string url = $"{API_URL}{voiceId}";
        
        var request = new UnityWebRequest(url, "POST");
        
        // JSON body
        string jsonBody = $@"{{
            ""text"": ""{text}"",
            ""model_id"": ""eleven_monolingual_v1"",
            ""voice_settings"": {{
                ""stability"": 0.5,
                ""similarity_boost"": 0.75
            }}
        }}";
        
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonBody);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerAudioClip(url, AudioType.MPEG);
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("xi-api-key", apiKey);
        
        yield return request.SendWebRequest();
        
        if (request.result == UnityWebRequest.Result.Success)
        {
            AudioClip clip = DownloadHandlerAudioClip.GetContent(request);
            
            // 2. Play audio
            audioSource.clip = clip;
            audioSource.Play();
            
            // 3. Start lip sync
            if (lipSync != null)
            {
                lipSync.StartLipSync(clip);
            }
            
            // 4. Wait for completion
            yield return new WaitForSeconds(clip.length);
            
            onComplete?.Invoke();
        }
        else
        {
            Debug.LogError($"ElevenLabs API Error: {request.error}");
            onComplete?.Invoke();
        }
    }
}

