using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

[RequireComponent(typeof(KellyTalkTest))]
public class KellyTTSClient : MonoBehaviour
{
    [TextArea(3, 6)]
    public string textToSpeak = "Hi! I'm Kelly.";
    public int learnerAge = 35;

    private KellyTalkTest talkTest;

    private const string TtsEndpoint = "http://localhost:3000/api/voice/tts";

    void Awake()
    {
        talkTest = GetComponent<KellyTalkTest>();
    }

    void Update()
    {
        // Press E to trigger ElevenLabs TTS
        bool ePressed = false;
        
        #if ENABLE_INPUT_SYSTEM
        var keyboard = UnityEngine.InputSystem.Keyboard.current;
        if (keyboard != null)
        {
            ePressed = keyboard.eKey.wasPressedThisFrame;
        }
        #else
        ePressed = Input.GetKeyDown(KeyCode.E);
        #endif
        
        if (ePressed)
        {
            Debug.Log("[KellyTTSClient] E key pressed - triggering ElevenLabs TTS");
            Speak();
        }
    }
    
    void Start()
    {
        Debug.Log("[KellyTTSClient] Component initialized. Press E or click button to test ElevenLabs TTS.");
    }

    [ContextMenu("Speak (ElevenLabs)")]
    public void Speak()
    {
        StartCoroutine(RequestAndPlay());
    }
    
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 220, 320, 120));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label("Kelly TTS Client (ElevenLabs)", GUI.skin.label);
        GUILayout.Space(5);
        
        GUILayout.Label("Press E key - Speak via ElevenLabs");
        GUILayout.Label($"Text: \"{textToSpeak}\"");
        
        if (GUILayout.Button("Speak Now (ElevenLabs)", GUILayout.Height(30)))
        {
            Debug.Log("[KellyTTSClient] Button clicked - triggering ElevenLabs TTS");
            Speak();
        }
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }

    IEnumerator RequestAndPlay()
    {
        if (talkTest == null)
        {
            Debug.LogError("[KellyTTSClient] KellyTalkTest component not found!");
            yield break;
        }

        Debug.Log($"[KellyTTSClient] Requesting TTS for: \"{textToSpeak.Substring(0, System.Math.Min(50, textToSpeak.Length))}...\"");

        var payload = JsonUtility.ToJson(new TtsRequest
        {
            age = learnerAge,
            text = textToSpeak
        });

        using var request = new UnityWebRequest(TtsEndpoint, UnityWebRequest.kHttpVerbPOST);
        var bodyRaw = System.Text.Encoding.UTF8.GetBytes(payload);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerAudioClip(TtsEndpoint, AudioType.MPEG);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"[KellyTTSClient] TTS request failed: {request.error}");
            Debug.LogError($"[KellyTTSClient] Response: {request.downloadHandler.text}");
            yield break;
        }

        var clip = DownloadHandlerAudioClip.GetContent(request);
        if (clip == null)
        {
            Debug.LogError("[KellyTTSClient] Failed to parse audio clip from TTS response.");
            yield break;
        }

        clip.name = "Kelly_TTS";

        Debug.Log($"[KellyTTSClient] Audio received! Length: {clip.length:F2}s");

        talkTest.testAudioClip = clip;
        talkTest.autoPlayOnStart = false;
        talkTest.PlayTestAudio();
    }

    [System.Serializable]
    struct TtsRequest
    {
        public int age;
        public string text;
    }
}

