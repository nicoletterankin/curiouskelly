using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

// Loads our extracted *_unity.json and drives Kelly CC5 blendshapes + head rotation.
//
// Intended project: digital-kelly/engines/Kelly_Engine_V2/onlykelly
// Notes:
// - WebGL build cannot read arbitrary local paths; use StreamingAssets (UnityWebRequest).
// - This drives CC5 visemes (V_*) + brows/eyes/smile from our extracted curves.
public class KellyAnimationPlayer : MonoBehaviour
{
    [Serializable]
    public class VisemeData
    {
        public float time;
        public float duration;
        public string viseme;
    }

    [Serializable]
    public class ExpressionFrame
    {
        public int frame;
        public float timestamp;
        public float mouthOpen;
        public float mouthWidth;
        public float smile;
        public float leftEyeOpen;
        public float rightEyeOpen;
        public float leftBrowRaise;
        public float rightBrowRaise;
        public float headYaw;
        public float headPitch;
        public float headRoll;
    }

    [Serializable]
    public class KellyAnimationData
    {
        public string clipName;
        public float duration;
        public float fps;
        public List<VisemeData> visemes;
        public List<ExpressionFrame> expressions;
    }

    [Header("References")]
    public ARKitBlendshapeController blendshapes;
    public Transform headBone;
    public AudioSource audioSource;

    [Header("StreamingAssets paths (relative)")]
    [Tooltip("e.g. kelly-motion/day_001_scientist_adult_unity.json")]
    public string animationJsonPath = "";

    [Tooltip("e.g. kelly-motion/day_001_scientist_adult.wav")]
    public string audioPath = "";

    [Header("Tuning")]
    public bool playOnStart = true;
    [Range(0f, 1f)] public float visemeStrength = 1.0f;
    [Range(0f, 1f)] public float expressionStrength = 1.0f;
    [Range(0f, 1f)] public float headStrength = 1.0f;
    [Range(0f, 1f)] public float blinkFromEyeOpenStrength = 0.9f;

    private KellyAnimationData animData;
    private float playbackTime;
    private bool isPlaying;
    private string activeCc5Viseme = "V_None";

    // Our extractor emits Unity viseme names like "viseme_PP". Map those to CC5 viseme blendshapes (V_*).
    private readonly Dictionary<string, string> visemeToCc5 = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
    {
        { "viseme_sil", "V_None" },
        { "viseme_aa",  "V_Open" },        // open vowel
        { "viseme_PP",  "V_Explosive" },   // p/b/m
        { "viseme_CH",  "V_Affricate" },   // ch/j
        { "viseme_DD",  "V_Dental_Lip" },  // d/t/th-ish
        { "viseme_E",   "V_Wide" },        // e
        { "viseme_FF",  "V_Dental_Lip" },  // f/v
        { "viseme_I",   "V_Wide" },        // i
        { "viseme_O",   "V_Tight_O" },     // o
    };

    private static readonly string[] AllCc5Visemes =
    {
        "V_None",
        "V_Open",
        "V_Explosive",
        "V_Dental_Lip",
        "V_Tight_O",
        "V_Tight",
        "V_Wide",
        "V_Affricate",
        "V_Lip_Open",
    };

    private void Awake()
    {
        if (audioSource == null) audioSource = GetComponent<AudioSource>();
        if (blendshapes == null) blendshapes = GetComponentInChildren<ARKitBlendshapeController>();
        if (blendshapes != null) blendshapes.InitializeBlendshapeMap();

        if (headBone == null)
        {
            headBone = FindDeepChild(transform, "CC_Base_Head") ??
                       FindDeepChild(transform, "CC_Base_HeadBone") ??
                       FindDeepChild(transform, "Head") ??
                       FindDeepChild(transform, "head");
        }
    }

    private void Start()
    {
        // Auto-play for quick end-to-end testing (especially in WebGL).
        if (playOnStart && !string.IsNullOrWhiteSpace(animationJsonPath))
        {
            LoadAndPlayFromInspectorPaths();
        }
    }

    public void LoadAndPlayFromInspectorPaths()
    {
        if (string.IsNullOrWhiteSpace(animationJsonPath))
        {
            Debug.LogError("[KellyAnimationPlayer] Missing animationJsonPath");
            return;
        }
        StartCoroutine(LoadAndPlay(animationJsonPath, audioPath));
    }

    // Can be called from JS via SendMessage in WebGL build, if desired:
    // unityInstance.SendMessage("kelly_fbx_v4", "LoadAndPlayFromStreamingAssets", "kelly-motion/day_001_scientist_adult_unity.json");
    public void LoadAndPlayFromStreamingAssets(string jsonRelativePath)
    {
        StartCoroutine(LoadAndPlay(jsonRelativePath, audioPath));
    }

    private IEnumerator LoadAndPlay(string jsonRelativePath, string audioRelativePath)
    {
        // Stop current
        Stop();

        // Load JSON
        string jsonUrl = CombineStreamingAssetsUrl(jsonRelativePath);
        using (var req = UnityWebRequest.Get(jsonUrl))
        {
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("[KellyAnimationPlayer] Failed to load JSON: " + req.error + " url=" + jsonUrl);
                yield break;
            }

            string json = req.downloadHandler.text;
            animData = JsonUtility.FromJson<KellyAnimationData>(json);
        }

        if (animData == null || animData.expressions == null || animData.expressions.Count == 0)
        {
            Debug.LogError("[KellyAnimationPlayer] Invalid animData (missing expressions).");
            yield break;
        }

        // Load audio (optional)
        if (!string.IsNullOrWhiteSpace(audioRelativePath))
        {
            string audioUrl = CombineStreamingAssetsUrl(audioRelativePath);
            using (var req = UnityWebRequestMultimedia.GetAudioClip(audioUrl, AudioType.WAV))
            {
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                {
                    var clip = DownloadHandlerAudioClip.GetContent(req);
                    audioSource.clip = clip;
                }
                else
                {
                    Debug.LogWarning("[KellyAnimationPlayer] Failed to load audio: " + req.error + " url=" + audioUrl);
                }
            }
        }

        Play();
    }

    public void Play()
    {
        if (animData == null)
        {
            Debug.LogError("[KellyAnimationPlayer] No animData loaded.");
            return;
        }

        playbackTime = 0f;
        isPlaying = true;

        if (audioSource != null && audioSource.clip != null)
        {
            audioSource.time = 0f;
            audioSource.Play();
        }
    }

    public void Stop()
    {
        isPlaying = false;
        playbackTime = 0f;

        if (audioSource != null) audioSource.Stop();
        ResetAllVisemes();
    }

    private void Update()
    {
        if (!isPlaying || animData == null) return;

        playbackTime += Time.deltaTime;
        if (playbackTime >= animData.duration)
        {
            Stop();
            return;
        }

        ApplyAtTime(playbackTime);
    }

    private void ApplyAtTime(float t)
    {
        ApplyViseme(t);
        ApplyExpression(t);
    }

    private void ApplyViseme(float time)
    {
        if (blendshapes == null || animData.visemes == null) return;

        // Find active viseme cue (linear scan; lists are short).
        VisemeData active = null;
        for (int i = 0; i < animData.visemes.Count; i++)
        {
            var v = animData.visemes[i];
            if (time >= v.time && time < (v.time + v.duration))
            {
                active = v;
                break;
            }
        }

        ResetAllVisemes();
        if (active == null || string.IsNullOrWhiteSpace(active.viseme)) return;

        if (!visemeToCc5.TryGetValue(active.viseme, out var cc5))
            cc5 = "V_None";

        activeCc5Viseme = cc5;

        // Silence: keep all visemes at 0 (neutral)
        if (string.Equals(activeCc5Viseme, "V_None", StringComparison.OrdinalIgnoreCase))
            return;

        blendshapes.SetBlendshape(activeCc5Viseme, Mathf.Clamp01(visemeStrength) * 100f);
    }

    private void ResetAllVisemes()
    {
        if (blendshapes == null) return;
        for (int i = 0; i < AllCc5Visemes.Length; i++)
            blendshapes.SetBlendshape(AllCc5Visemes[i], 0f);
    }

    private void ApplyExpression(float time)
    {
        if (blendshapes == null || animData.expressions == null || animData.expressions.Count == 0) return;

        // Nearest frame by fps
        int idx = Mathf.Clamp(Mathf.FloorToInt(time * animData.fps), 0, animData.expressions.Count - 1);
        var fr = animData.expressions[idx];

        float e = Mathf.Clamp01(expressionStrength);

        // Smile -> CC5 mouth smile
        float smile = Mathf.Clamp01(fr.smile) * e * 100f;
        blendshapes.SetBlendshape("Mouth_Smile_L", smile);
        blendshapes.SetBlendshape("Mouth_Smile_R", smile);

        // Brows
        float browL = Mathf.Clamp01(fr.leftBrowRaise) * e * 100f;
        float browR = Mathf.Clamp01(fr.rightBrowRaise) * e * 100f;
        blendshapes.SetBlendshape("Brow_Raise_Inner_L", browL);
        blendshapes.SetBlendshape("Brow_Raise_Inner_R", browR);

        // Eye open -> blink (inverse)
        float leftBlink = Mathf.Clamp01(1f - fr.leftEyeOpen) * blinkFromEyeOpenStrength * e * 100f;
        float rightBlink = Mathf.Clamp01(1f - fr.rightEyeOpen) * blinkFromEyeOpenStrength * e * 100f;
        blendshapes.SetBlendshape("Eye_Blink_L", leftBlink);
        blendshapes.SetBlendshape("Eye_Blink_R", rightBlink);

        // Optional: use mouthOpen/mouthWidth as subtle modifiers (helps when visemes are coarse).
        // We'll nudge V_Open and V_Wide a little, without overriding the active viseme.
        float mouthOpen = Mathf.Clamp01(fr.mouthOpen) * 35f * e;
        float mouthWidth = Mathf.Clamp01(fr.mouthWidth) * 20f * e;
        if (!string.Equals(activeCc5Viseme, "V_Open", StringComparison.OrdinalIgnoreCase))
            blendshapes.SetBlendshape("V_Open", Mathf.Max(0f, mouthOpen));
        if (!string.Equals(activeCc5Viseme, "V_Wide", StringComparison.OrdinalIgnoreCase))
            blendshapes.SetBlendshape("V_Wide", Mathf.Max(0f, mouthWidth));

        // Head rotation (degrees) -> bone local rotation.
        if (headBone != null)
        {
            float h = Mathf.Clamp01(headStrength);
            // Our extractor outputs yaw/pitch/roll already in degrees (rough).
            headBone.localRotation = Quaternion.Euler(fr.headPitch * h, fr.headYaw * h, fr.headRoll * h);
        }
    }

    private static string CombineStreamingAssetsUrl(string relativePath)
    {
        // Handles WebGL/desktop path formats.
        string basePath = Application.streamingAssetsPath;
        if (string.IsNullOrEmpty(relativePath)) return basePath;
        relativePath = relativePath.Replace("\\", "/").TrimStart('/');
        if (basePath.EndsWith("/")) return basePath + relativePath;
        return basePath + "/" + relativePath;
    }

    private static Transform FindDeepChild(Transform parent, string name)
    {
        if (parent == null) return null;
        var queue = new Queue<Transform>();
        queue.Enqueue(parent);
        while (queue.Count > 0)
        {
            var t = queue.Dequeue();
            if (string.Equals(t.name, name, StringComparison.OrdinalIgnoreCase))
                return t;
            for (int i = 0; i < t.childCount; i++)
                queue.Enqueue(t.GetChild(i));
        }
        return null;
    }
}


