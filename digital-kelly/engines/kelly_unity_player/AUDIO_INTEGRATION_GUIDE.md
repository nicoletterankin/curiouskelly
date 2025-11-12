# Unity Audio Integration Guide

## 🎙️ **Testing Lesson Audio in Unity**

This guide shows how to import and test the generated lesson audio files in Unity.

---

## 📁 **Step 1: Copy Audio Files to Unity** (5 minutes)

### **Create Audio Folder Structure**

```
Unity Project/
└── Assets/
    └── Resources/
        └── Audio/
            └── Lessons/
                ├── water-cycle/
                │   ├── 2-5-welcome.mp3
                │   ├── 2-5-mainContent.mp3
                │   ├── 2-5-wisdomMoment.mp3
                │   ├── 6-12-welcome.mp3
                │   ├── ... (all 18 files)
                │   └── 61-102-wisdomMoment.mp3
                └── leaves-change-color/
                    └── (when ready)
```

### **Copy Files (Windows)**

```powershell
# From: curious-kellly\backend\config\audio\water-cycle\
# To: Unity project Assets\Resources\Audio\Lessons\water-cycle\

# Create folder in Unity project
New-Item -ItemType Directory -Force -Path "digital-kelly\engines\kelly_unity_player\Assets\Resources\Audio\Lessons\water-cycle"

# Copy all MP3 files
Copy-Item "curious-kellly\backend\config\audio\water-cycle\*.mp3" `
          "digital-kelly\engines\kelly_unity_player\Assets\Resources\Audio\Lessons\water-cycle\"
```

---

## 🎬 **Step 2: Set Up Audio in Unity** (10 minutes)

### **Import Settings**

1. Open Unity project: `digital-kelly/engines/kelly_unity_player`
2. Select all MP3 files in Project window
3. In Inspector, set:
   - **Load Type:** Streaming (for large files) or Compressed in Memory
   - **Preload Audio Data:** ✅ Check (for instant playback)
   - **Compression Format:** Vorbis
   - **Quality:** 100% (for best quality)
   - **Sample Rate Setting:** Preserve Sample Rate

4. Click **Apply**

### **Add Components to Kelly**

1. Select Kelly avatar in Hierarchy
2. Add Component → **LessonAudioPlayer** (script we created)
3. Add Component → **Audio Source** (if not already present)
4. In **LessonAudioPlayer** Inspector:
   - **Lesson Id:** water-cycle
   - **Age Group:** 18-35
   - **Auto Play On Load:** ✅ (for testing)

---

## ▶️ **Step 3: Test Playback** (5 minutes)

### **Method 1: Play Mode (Automatic)**

1. Click **Play** ▶️ in Unity
2. Audio should start playing automatically
3. Check Console for logs:
   ```
   [LessonAudioPlayer] Playing complete lesson: water-cycle
   [LessonAudioPlayer] Started: Welcome (5.2s)
   ```

### **Method 2: Play Mode (Manual Control)**

1. Click **Play** ▶️
2. In Inspector, click:
   - **Play Complete Lesson** button
   - Or **Play Section** → Welcome/MainContent/WisdomMoment

### **Method 3: Test Script**

Create a test UI button:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class AudioTestUI : MonoBehaviour
{
    public LessonAudioPlayer audioPlayer;
    public Button playButton;
    public Button pauseButton;
    public Button stopButton;
    public Text statusText;
    
    void Start()
    {
        playButton.onClick.AddListener(() => {
            audioPlayer.PlayCompleteLesson();
        });
        
        pauseButton.onClick.AddListener(() => {
            audioPlayer.Pause();
        });
        
        stopButton.onClick.AddListener(() => {
            audioPlayer.Stop();
        });
        
        audioPlayer.OnSectionStarted += (section) => {
            statusText.text = $"Playing: {section}";
        };
        
        audioPlayer.OnProgressUpdate += (progress) => {
            statusText.text = $"Progress: {(progress * 100):F0}%";
        };
    }
}
```

---

## 🔊 **Step 4: Test All Age Variants** (10 minutes)

Test each Kelly age to verify voice quality:

### **Quick Test Script**

```csharp
using UnityEngine;

public class AudioAgeTest : MonoBehaviour
{
    private LessonAudioPlayer audioPlayer;
    private string[] ageGroups = { "2-5", "6-12", "13-17", "18-35", "36-60", "61-102" };
    private int currentAgeIndex = 0;
    
    void Start()
    {
        audioPlayer = GetComponent<LessonAudioPlayer>();
    }
    
    void Update()
    {
        // Press Space to test next age
        if (Input.GetKeyDown(KeyCode.Space))
        {
            TestNextAge();
        }
    }
    
    void TestNextAge()
    {
        string ageGroup = ageGroups[currentAgeIndex];
        Debug.Log($"Testing age group: {ageGroup}");
        
        audioPlayer.LoadLessonAudio("water-cycle", ageGroup);
        audioPlayer.PlaySection(LessonSection.Welcome);
        
        currentAgeIndex = (currentAgeIndex + 1) % ageGroups.Length;
    }
}
```

**Usage:**
1. Attach script to Kelly
2. Play in Unity
3. Press **Spacebar** to cycle through ages
4. Listen to each Kelly voice!

---

## 🎨 **Step 5: Sync with Avatar** (Optional, 20 minutes)

Connect audio to Kelly's lip-sync:

### **Basic Lip-Sync (Volume-Based)**

```csharp
using UnityEngine;

public class SimpleLipSync : MonoBehaviour
{
    public SkinnedMeshRenderer headRenderer;
    public AudioSource audioSource;
    public int jawOpenBlendshapeIndex = 0;
    public float sensitivity = 100f;
    
    private float[] samples = new float[64];
    
    void Update()
    {
        if (audioSource.isPlaying)
        {
            audioSource.GetOutputData(samples, 0);
            
            float volume = 0f;
            foreach (float sample in samples)
            {
                volume += Mathf.Abs(sample);
            }
            volume /= samples.Length;
            
            float jawOpen = Mathf.Clamp01(volume * sensitivity);
            headRenderer.SetBlendShapeWeight(jawOpenBlendshapeIndex, jawOpen * 100f);
        }
        else
        {
            // Close mouth when not speaking
            headRenderer.SetBlendShapeWeight(jawOpenBlendshapeIndex, 0f);
        }
    }
}
```

**For proper viseme-based lip-sync**, you'll need:
1. Viseme data from Audio2Face
2. BlendshapeDriver60fps integration
3. Timing synchronization

---

## 🧪 **Troubleshooting**

### **Issue: No audio plays**

**Check:**
1. MP3 files copied to correct location?
2. Audio Source component present?
3. Volume > 0?
4. Mute in Unity disabled?
5. System audio working?

**Debug:**
```csharp
Debug.Log($"Audio clip loaded: {audioPlayer.welcomeClip != null}");
Debug.Log($"Audio Source: {audioPlayer.audioSource != null}");
Debug.Log($"Volume: {audioPlayer.audioSource.volume}");
```

### **Issue: Audio clips null**

**Solution:** Make sure files are in `Assets/Resources/Audio/Lessons/` folder

Unity's `Resources.Load()` only works with files in a `Resources` folder!

### **Issue: Poor audio quality**

**Solution:** Adjust import settings:
- **Compression Format:** PCM (uncompressed) for best quality
- **Quality:** 100%
- **Load Type:** Decompress On Load (for better performance)

### **Issue: Lag when starting playback**

**Solution:**
- Enable **Preload Audio Data** in import settings
- Use **Decompress On Load** for short clips
- Use **Streaming** for long clips (>1 minute)

---

## 📊 **Expected Results**

### **Age 3 (Toddler) - welcome.mp3**
- **Duration:** ~2 seconds
- **Voice:** High-pitched, friendly (Nova)
- **Content:** "Hi friend! Do you see water?..."

### **Age 27 (Adult) - mainContent.mp3**
- **Duration:** ~30 seconds
- **Voice:** Clear, professional (Shimmer)
- **Content:** "The hydrological cycle represents..."

### **Age 82 (Elder) - wisdomMoment.mp3**
- **Duration:** ~8 seconds
- **Voice:** Expressive, profound (Fable)
- **Content:** "Water returns to the sea, and we return to water..."

---

## 🚀 **Next Steps**

### **After Basic Playback Works:**

1. **Add UI Controls**
   - Play/Pause/Stop buttons
   - Progress slider
   - Age selector

2. **Sync with Avatar**
   - Integrate with BlendshapeDriver60fps
   - Add viseme data
   - Test lip-sync accuracy

3. **Add Flutter Communication**
   - Send playback events to Flutter
   - Receive lesson start commands
   - Sync audio with lesson phases

4. **Test Complete Flow**
   - User starts lesson in Flutter
   - Unity receives message
   - Kelly speaks with audio
   - Avatar lip-syncs perfectly
   - Lesson completes

---

## 📁 **Files Created**

✅ `LessonAudioPlayer.cs` - Audio playback controller  
✅ `AUDIO_INTEGRATION_GUIDE.md` - This guide  

**To create:**
- `SimpleLipSync.cs` - Basic volume-based lip-sync
- `AudioTestUI.cs` - Test UI controls
- `AudioAgeTest.cs` - Age variant tester

---

## 🎉 **Success Criteria**

✅ All 18 audio files import without errors  
✅ Audio plays in Unity Play Mode  
✅ Can switch between age groups  
✅ Volume and controls work  
✅ No lag or stuttering  
✅ Voice quality is clear and natural  

**When these all work, you're ready to integrate with the full lesson system!** 🚀

---

**Questions?** Check Unity Console for debug logs or re-check file paths in Resources folder.














