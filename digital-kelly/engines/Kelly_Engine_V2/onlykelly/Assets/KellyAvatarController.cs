using UnityEngine;
using System.Collections.Generic;

public class KellyAvatarController : MonoBehaviour
{
    [Header("Face Configuration")]
    public SkinnedMeshRenderer faceMesh; // Drag her face here
    public int visemeMultiplier = 100;
    
    [Header("Animation")]
    public Animator animator;
    
    [Header("State")]
    private string currentExpression = "curious";
    private bool isSpeaking = false;
    private Coroutine lipSyncCoroutine;

    // Map standard visemes to CC4/iClone blendshape names
    private Dictionary<string, string> visemeMap = new Dictionary<string, string>
    {
        {"sil", "V_Explosive"}, // Silence/B/P/M
        {"PP", "V_Explosive"},
        {"FF", "V_Dental_Lip"},
        {"TH", "V_Tight_O"},
        {"DD", "V_Dental_Lip"},
        {"kk", "V_Tight_O"},
        {"CH", "V_Tight_O"},
        {"SS", "V_Dental_Lip"},
        {"nn", "V_Dental_Lip"},
        {"RR", "V_Tight_O"},
        {"aa", "V_Wide"},
        {"E", "V_Dental_Lip"},
        {"ih", "V_Wide"},
        {"oh", "V_Tight_O"},
        {"ou", "V_Tight_O"}
    };
    
    // Expression blendshape mappings (adjust names for your model)
    private Dictionary<string, Dictionary<string, float>> expressionMap = new Dictionary<string, Dictionary<string, float>>
    {
        {"curious", new Dictionary<string, float> { {"A01_Brow_Inner_Up", 30f}, {"A05_Upper_Lid_Raiser", 20f} }},
        {"explaining", new Dictionary<string, float> { {"A01_Brow_Inner_Up", 40f}, {"A06_Cheek_Raiser", 30f} }},
        {"listening", new Dictionary<string, float> { {"A02_Brow_Outer_Up_L", 20f}, {"A02_Brow_Outer_Up_R", 20f} }},
        {"wisdom", new Dictionary<string, float> { {"A12_Lip_Corner_Puller", 40f}, {"A06_Cheek_Raiser", 50f} }},
        {"celebrating", new Dictionary<string, float> { {"A12_Lip_Corner_Puller", 80f}, {"A06_Cheek_Raiser", 70f}, {"A01_Brow_Inner_Up", 50f} }}
    };

    void Start()
    {
        // Get animator if not assigned
        if (animator == null)
        {
            animator = GetComponent<Animator>();
        }
        Debug.Log("[KellyAvatarController] Initialized");
    }
    
    // Called from JavaScript: SetExpression("curious")
    public void SetExpression(string expression)
    {
        Debug.Log($"[KellyAvatarController] SetExpression: {expression}");
        
        if (faceMesh == null) 
        {
            Debug.LogWarning("[KellyAvatarController] faceMesh not assigned!");
            return;
        }
        
        // Reset previous expression
        if (expressionMap.ContainsKey(currentExpression))
        {
            foreach (var kvp in expressionMap[currentExpression])
            {
                int index = faceMesh.sharedMesh.GetBlendShapeIndex(kvp.Key);
                if (index != -1)
                {
                    faceMesh.SetBlendShapeWeight(index, 0f);
                }
            }
        }
        
        // Apply new expression
        currentExpression = expression;
        if (expressionMap.ContainsKey(expression))
        {
            foreach (var kvp in expressionMap[expression])
            {
                int index = faceMesh.sharedMesh.GetBlendShapeIndex(kvp.Key);
                if (index != -1)
                {
                    faceMesh.SetBlendShapeWeight(index, kvp.Value);
                }
            }
        }
    }
    
    // Called from JavaScript: StartLipSync("Hello there")
    public void StartLipSync(string text)
    {
        Debug.Log($"[KellyAvatarController] StartLipSync: {text}");
        isSpeaking = true;
        
        // You can integrate with audio/lip sync system here
        // For now, just set a "talking" animation state
        if (animator != null)
        {
            animator.SetBool("IsTalking", true);
        }
    }
    
    // Called from JavaScript: StopLipSync()
    public void StopLipSync()
    {
        Debug.Log("[KellyAvatarController] StopLipSync");
        isSpeaking = false;
        
        // Reset to neutral mouth
        ResetVisemes();
        
        if (animator != null)
        {
            animator.SetBool("IsTalking", false);
        }
    }
    
    // Called from JavaScript: PlayAnimation("celebrate")
    public void PlayAnimation(string animationName)
    {
        Debug.Log($"[KellyAvatarController] PlayAnimation: {animationName}");
        
        if (animator != null)
        {
            animator.SetTrigger(animationName);
        }
    }
    
    private void ResetVisemes()
    {
        if (faceMesh == null) return;
        
        foreach (var kvp in visemeMap)
        {
            int index = faceMesh.sharedMesh.GetBlendShapeIndex(kvp.Value);
            if (index != -1)
            {
                faceMesh.SetBlendShapeWeight(index, 0f);
            }
        }
    }

    public void SetViseme(string visemeName, float weight)
    {
        if (faceMesh == null) return;

        if (visemeMap.ContainsKey(visemeName))
        {
            string blendShapeName = visemeMap[visemeName];
            int index = faceMesh.sharedMesh.GetBlendShapeIndex(blendShapeName);
            if (index != -1)
            {
                faceMesh.SetBlendShapeWeight(index, weight * visemeMultiplier);
            }
        }
    }

    // Called by the Website via SendMessage
    public void ProcessViseme(string json)
    {
        // Format: "aa:0.5"
        string[] parts = json.Split(':');
        if (parts.Length == 2)
        {
            if (float.TryParse(parts[1], out float w))
            {
                SetViseme(parts[0], w);
            }
        }
    }
}