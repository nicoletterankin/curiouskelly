using UnityEngine;
using System.Collections;

/// <summary>
/// Kelly Chair Controller - Master of Ceremonies Mode
/// 
/// This controller extends the base Kelly avatar system to support the director's chair pose.
/// It implements the state machine from PRD_KELLY_UNIFIED_EXPERIENCE.md where Kelly is
/// ALWAYS visible as the MC for lessons, payments, settings, and celebrations.
/// 
/// Modes: ONBOARDING, TEACHING, CELEBRATING, UPGRADING, WAITING, THINKING, SETTINGS
/// </summary>
public class KellyChairController : MonoBehaviour
{
    public enum KellyMode
    {
        ONBOARDING,     // "Welcome! Let me show you around..."
        TEACHING,       // Active lesson delivery
        CELEBRATING,    // "You did it!" - lesson complete
        UPGRADING,      // "Ready to unlock more?" - paywall
        WAITING,        // Idle, breathing, occasional blink
        THINKING,       // User is processing
        SETTINGS        // "Let me help you customize..."
    }

    [Header("References")]
    public KellyAvatarController avatarController;
    public KellyAnimationPlayer animationPlayer;
    public SkinnedMeshRenderer faceMesh;
    public Transform chairTransform;    // Reference to chair prop
    
    [Header("Current State")]
    public KellyMode currentMode = KellyMode.WAITING;
    
    [Header("Idle Animation Settings")]
    [Range(0.5f, 3f)] public float breathingSpeed = 1.5f;
    [Range(0.02f, 0.1f)] public float breathingAmount = 0.05f;
    [Range(3f, 8f)] public float blinkInterval = 5f;
    [Range(0.1f, 0.3f)] public float blinkDuration = 0.15f;
    
    [Header("State Expressions")]
    [SerializeField] private string onboardingExpression = "curious";
    [SerializeField] private string teachingExpression = "explaining";
    [SerializeField] private string celebratingExpression = "celebrating";
    [SerializeField] private string upgradingExpression = "listening";
    [SerializeField] private string waitingExpression = "curious";
    [SerializeField] private string thinkingExpression = "wisdom";
    [SerializeField] private string settingsExpression = "listening";
    
    // Idle animation state
    private float breathingPhase = 0f;
    private float blinkTimer = 0f;
    private bool isBlinking = false;
    private Coroutine celebrationCoroutine;
    
    // Chair position (seated)
    private Vector3 basePosition;
    private Quaternion baseRotation;

    void Start()
    {
        // Cache base transforms
        basePosition = transform.localPosition;
        baseRotation = transform.localRotation;
        
        // Auto-find references if not assigned
        if (avatarController == null)
            avatarController = GetComponent<KellyAvatarController>();
        if (animationPlayer == null)
            animationPlayer = GetComponent<KellyAnimationPlayer>();
        if (faceMesh == null)
            faceMesh = GetComponentInChildren<SkinnedMeshRenderer>();
            
        // Initialize blink timer with random offset
        blinkTimer = Random.Range(2f, blinkInterval);
        
        // Set initial state
        SetMode(KellyMode.WAITING);
        
        Debug.Log("[KellyChairController] Initialized - Kelly is in the building!");
    }

    void Update()
    {
        // Only apply idle animations when not actively speaking/teaching
        if (currentMode == KellyMode.WAITING || 
            currentMode == KellyMode.THINKING || 
            currentMode == KellyMode.SETTINGS)
        {
            ApplyIdleAnimations();
        }
    }

    #region Mode Control (Called from JavaScript/Web)

    /// <summary>
    /// Set Kelly's mode - called from JavaScript via SendMessage
    /// Format: "TEACHING", "CELEBRATING", etc.
    /// </summary>
    public void SetModeFromJS(string modeName)
    {
        if (System.Enum.TryParse<KellyMode>(modeName, true, out KellyMode mode))
        {
            SetMode(mode);
        }
        else
        {
            Debug.LogWarning($"[KellyChairController] Unknown mode: {modeName}");
        }
    }

    public void SetMode(KellyMode mode)
    {
        if (currentMode == mode) return;
        
        KellyMode previousMode = currentMode;
        currentMode = mode;
        
        Debug.Log($"[KellyChairController] Mode: {previousMode} → {mode}");
        
        // Stop any celebration in progress
        if (celebrationCoroutine != null)
        {
            StopCoroutine(celebrationCoroutine);
            celebrationCoroutine = null;
        }
        
        // Apply mode-specific behavior
        switch (mode)
        {
            case KellyMode.ONBOARDING:
                avatarController?.SetExpression(onboardingExpression);
                break;
                
            case KellyMode.TEACHING:
                avatarController?.SetExpression(teachingExpression);
                break;
                
            case KellyMode.CELEBRATING:
                celebrationCoroutine = StartCoroutine(CelebrationSequence());
                break;
                
            case KellyMode.UPGRADING:
                avatarController?.SetExpression(upgradingExpression);
                break;
                
            case KellyMode.WAITING:
                avatarController?.SetExpression(waitingExpression);
                break;
                
            case KellyMode.THINKING:
                avatarController?.SetExpression(thinkingExpression);
                break;
                
            case KellyMode.SETTINGS:
                avatarController?.SetExpression(settingsExpression);
                break;
        }
        
        // Notify JavaScript of mode change
        NotifyModeChange(mode.ToString());
    }

    #endregion

    #region Idle Animations

    private void ApplyIdleAnimations()
    {
        ApplyBreathing();
        ApplyBlink();
    }

    private void ApplyBreathing()
    {
        breathingPhase += Time.deltaTime * breathingSpeed;
        if (breathingPhase > Mathf.PI * 2f)
            breathingPhase -= Mathf.PI * 2f;
            
        // Subtle vertical float
        float breathOffset = Mathf.Sin(breathingPhase) * breathingAmount;
        transform.localPosition = basePosition + Vector3.up * breathOffset;
        
        // Subtle shoulder movement on face mesh (if available)
        // This gives life to the static seated pose
    }

    private void ApplyBlink()
    {
        blinkTimer -= Time.deltaTime;
        
        if (blinkTimer <= 0 && !isBlinking)
        {
            StartCoroutine(BlinkOnce());
        }
    }

    private IEnumerator BlinkOnce()
    {
        isBlinking = true;
        
        if (faceMesh != null)
        {
            // Find blink blendshapes
            int leftBlinkIdx = faceMesh.sharedMesh.GetBlendShapeIndex("Eye_Blink_L");
            int rightBlinkIdx = faceMesh.sharedMesh.GetBlendShapeIndex("Eye_Blink_R");
            
            if (leftBlinkIdx >= 0 && rightBlinkIdx >= 0)
            {
                // Close eyes
                float t = 0;
                while (t < blinkDuration / 2f)
                {
                    t += Time.deltaTime;
                    float weight = Mathf.Lerp(0, 100, t / (blinkDuration / 2f));
                    faceMesh.SetBlendShapeWeight(leftBlinkIdx, weight);
                    faceMesh.SetBlendShapeWeight(rightBlinkIdx, weight);
                    yield return null;
                }
                
                // Open eyes
                t = 0;
                while (t < blinkDuration / 2f)
                {
                    t += Time.deltaTime;
                    float weight = Mathf.Lerp(100, 0, t / (blinkDuration / 2f));
                    faceMesh.SetBlendShapeWeight(leftBlinkIdx, weight);
                    faceMesh.SetBlendShapeWeight(rightBlinkIdx, weight);
                    yield return null;
                }
            }
        }
        
        // Reset timer with variation
        blinkTimer = blinkInterval + Random.Range(-1f, 2f);
        isBlinking = false;
    }

    #endregion

    #region Celebration Sequence

    private IEnumerator CelebrationSequence()
    {
        avatarController?.SetExpression(celebratingExpression);
        
        // Optional: trigger celebration animation
        avatarController?.PlayAnimation("celebrate");
        
        // Hold celebration for 3-5 seconds
        yield return new WaitForSeconds(3.5f);
        
        // Return to waiting mode
        SetMode(KellyMode.WAITING);
    }

    #endregion

    #region JavaScript Communication

    /// <summary>
    /// Called from web to start teaching (audio playing)
    /// </summary>
    public void OnAudioStart()
    {
        SetMode(KellyMode.TEACHING);
    }

    /// <summary>
    /// Called from web when audio stops
    /// </summary>
    public void OnAudioEnd()
    {
        SetMode(KellyMode.WAITING);
    }

    /// <summary>
    /// Called from web when lesson completes
    /// </summary>
    public void OnLessonComplete()
    {
        SetMode(KellyMode.CELEBRATING);
    }

    /// <summary>
    /// Called from web when paywall shows
    /// </summary>
    public void OnPaywallShow()
    {
        SetMode(KellyMode.UPGRADING);
    }

    /// <summary>
    /// Send mode change event to JavaScript
    /// </summary>
    private void NotifyModeChange(string mode)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        Application.ExternalCall("onKellyModeChange", mode);
#endif
        Debug.Log($"[KellyChairController] NotifyModeChange: {mode}");
    }

    #endregion

    #region Quick Tests (Editor)

    [ContextMenu("Test: Set TEACHING Mode")]
    private void TestTeaching() => SetMode(KellyMode.TEACHING);

    [ContextMenu("Test: Set CELEBRATING Mode")]
    private void TestCelebrating() => SetMode(KellyMode.CELEBRATING);

    [ContextMenu("Test: Set WAITING Mode")]
    private void TestWaiting() => SetMode(KellyMode.WAITING);

    #endregion
}
