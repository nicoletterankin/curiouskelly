using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class KellyBridge : MonoBehaviour
{
    public BlendshapeDriver driver;

    // Called by Flutter: payload = "path/to/json|path/to/wav"
    public void LoadAndPlay(string payload)
    {
        var parts = payload.Split('|');
        var jsonPath = parts[0];
        var wavPath = parts[1];

        Debug.Log($"📥 KellyBridge: Received load request - JSON: {jsonPath}, WAV: {wavPath}");

        if (File.Exists(jsonPath))
        {
            string json = File.ReadAllText(jsonPath);
            driver.LoadRuntimeJson(json);
            Debug.Log("✅ KellyBridge: Loaded A2F data");
        }
        else
        {
            Debug.LogWarning($"⚠️ KellyBridge: JSON file not found: {jsonPath}");
        }

        if (File.Exists(wavPath))
        {
            StartCoroutine(LoadClipAndPlay(wavPath));
        }
        else
        {
            Debug.LogWarning($"⚠️ KellyBridge: WAV file not found: {wavPath}");
        }
    }

    private System.Collections.IEnumerator LoadClipAndPlay(string path)
    {
        Debug.Log($"🎵 KellyBridge: Loading audio from {path}");
        using var req = UnityWebRequestMultimedia.GetAudioClip("file://" + path, AudioType.WAV);
        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
        {
            var clip = DownloadHandlerAudioClip.GetContent(req);
            driver.SetAudioClip(clip);
            driver.PlaySynced(0.05);
            Debug.Log("✅ KellyBridge: Audio playing in sync");
        }
        else
        {
            Debug.LogError($"❌ KellyBridge: Failed to load audio: {req.error}");
        }
    }
}








