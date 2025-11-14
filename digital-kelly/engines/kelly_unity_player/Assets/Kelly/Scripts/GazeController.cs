using UnityEngine;
using System.Collections;

/// <summary>
/// Gaze Controller for natural eye movement
/// Implements gaze tracking with micro-saccades for realistic eye behavior
/// </summary>
public class GazeController : MonoBehaviour
{
    [Header("Eye Transforms")]
    public Transform leftEyeBone;
    public Transform rightEyeBone;
    
    [Header("Gaze Settings")]
    public float gazeSpeed = 3f;
    public float maxGazeAngle = 30f; // Maximum eye rotation in degrees
    
    [Header("Micro-Saccade Settings")]
    public bool enableMicroSaccades = true;
    public float saccadeFrequency = 3f; // 2-4 per second
    public float saccadeAmplitude = 2f; // Degrees
    public float saccadeDuration = 0.05f; // 50ms
    
    [Header("Target Settings")]
    public Transform defaultGazeTarget; // Usually camera
    
    // Current state
    private Vector3 currentGazeTarget;
    private GazeTargetType currentTargetType = GazeTargetType.Camera;
    private Quaternion leftEyeTargetRotation;
    private Quaternion rightEyeTargetRotation;
    private Quaternion leftEyeDefaultRotation;
    private Quaternion rightEyeDefaultRotation;
    
    // Micro-saccade state
    private float nextSaccadeTime = 0f;
    private Vector3 saccadeOffset = Vector3.zero;
    private bool isSaccading = false;
    
    // Performance
    private bool isInitialized = false;
    
    void Start()
    {
        Initialize();
    }
    
    void Initialize()
    {
        if (isInitialized) return;
        
        // Store default rotations
        if (leftEyeBone != null)
        {
            leftEyeDefaultRotation = leftEyeBone.localRotation;
        }
        
        if (rightEyeBone != null)
        {
            rightEyeDefaultRotation = rightEyeBone.localRotation;
        }
        
        // Set default target
        if (defaultGazeTarget == null)
        {
            defaultGazeTarget = Camera.main?.transform;
        }
        
        if (defaultGazeTarget != null)
        {
            currentGazeTarget = defaultGazeTarget.position;
        }
        
        isInitialized = true;
        Debug.Log("[Gaze Controller] Initialized");
    }
    
    void LateUpdate()
    {
        if (!isInitialized) return;
        
        // Update gaze target
        UpdateGazeDirection();
        
        // Apply micro-saccades
        if (enableMicroSaccades)
        {
            UpdateMicroSaccades();
        }
        
        // Apply rotations to eyes
        ApplyGazeRotation();
    }
    
    void UpdateGazeDirection()
    {
        // Calculate target direction
        Vector3 targetPosition = currentGazeTarget + saccadeOffset;
        
        // For left eye
        if (leftEyeBone != null)
        {
            Vector3 directionLeft = (targetPosition - leftEyeBone.position).normalized;
            Quaternion targetRotLeft = Quaternion.LookRotation(directionLeft);
            
            // Clamp rotation
            targetRotLeft = ClampRotation(targetRotLeft, leftEyeDefaultRotation, maxGazeAngle);
            leftEyeTargetRotation = targetRotLeft;
        }
        
        // For right eye
        if (rightEyeBone != null)
        {
            Vector3 directionRight = (targetPosition - rightEyeBone.position).normalized;
            Quaternion targetRotRight = Quaternion.LookRotation(directionRight);
            
            // Clamp rotation
            targetRotRight = ClampRotation(targetRotRight, rightEyeDefaultRotation, maxGazeAngle);
            rightEyeTargetRotation = targetRotRight;
        }
    }
    
    void UpdateMicroSaccades()
    {
        if (Time.time >= nextSaccadeTime && !isSaccading)
        {
            StartCoroutine(PerformMicroSaccade());
        }
    }
    
    IEnumerator PerformMicroSaccade()
    {
        isSaccading = true;
        
        // Random saccade direction
        Vector3 randomOffset = Random.insideUnitSphere * saccadeAmplitude * 0.01f;
        saccadeOffset = randomOffset;
        
        // Wait for saccade duration
        yield return new WaitForSeconds(saccadeDuration);
        
        // Return to center
        saccadeOffset = Vector3.zero;
        
        // Schedule next saccade
        float interval = 1f / saccadeFrequency;
        nextSaccadeTime = Time.time + interval + Random.Range(-0.1f, 0.1f);
        
        isSaccading = false;
    }
    
    void ApplyGazeRotation()
    {
        // Smoothly interpolate to target rotation
        if (leftEyeBone != null)
        {
            leftEyeBone.localRotation = Quaternion.Slerp(
                leftEyeBone.localRotation,
                leftEyeTargetRotation,
                gazeSpeed * Time.deltaTime
            );
        }
        
        if (rightEyeBone != null)
        {
            rightEyeBone.localRotation = Quaternion.Slerp(
                rightEyeBone.localRotation,
                rightEyeTargetRotation,
                gazeSpeed * Time.deltaTime
            );
        }
    }
    
    Quaternion ClampRotation(Quaternion rotation, Quaternion defaultRotation, float maxAngle)
    {
        // Calculate angle from default
        float angle = Quaternion.Angle(rotation, defaultRotation);
        
        if (angle > maxAngle)
        {
            // Clamp to max angle
            return Quaternion.RotateTowards(defaultRotation, rotation, maxAngle);
        }
        
        return rotation;
    }
    
    /// <summary>
    /// Set gaze target position (world space)
    /// </summary>
    public void SetGazeTarget(Vector3 worldPosition)
    {
        currentGazeTarget = worldPosition;
    }
    
    /// <summary>
    /// Set gaze target by type
    /// </summary>
    public void SetGazeTarget(GazeTargetType targetType)
    {
        currentTargetType = targetType;
        
        switch (targetType)
        {
            case GazeTargetType.Camera:
                if (defaultGazeTarget != null)
                {
                    currentGazeTarget = defaultGazeTarget.position;
                }
                break;
                
            case GazeTargetType.Left:
                currentGazeTarget = transform.position + transform.TransformDirection(Vector3.left * 2f);
                break;
                
            case GazeTargetType.Right:
                currentGazeTarget = transform.position + transform.TransformDirection(Vector3.right * 2f);
                break;
                
            case GazeTargetType.Up:
                currentGazeTarget = transform.position + transform.TransformDirection(Vector3.up * 2f);
                break;
                
            case GazeTargetType.Down:
                currentGazeTarget = transform.position + transform.TransformDirection(Vector3.down * 2f);
                break;
                
            case GazeTargetType.Content:
                // Look slightly down and forward (at content)
                currentGazeTarget = transform.position + transform.TransformDirection(new Vector3(0, -0.5f, 2f));
                break;
        }
    }
    
    /// <summary>
    /// Set gaze target from screen position (for touch interaction)
    /// </summary>
    public void SetGazeTargetFromScreen(Vector2 screenPosition)
    {
        Ray ray = Camera.main.ScreenPointToRay(screenPosition);
        currentGazeTarget = ray.GetPoint(2f); // 2 meters in front
    }
    
    /// <summary>
    /// Return to default gaze (camera)
    /// </summary>
    public void ResetGaze()
    {
        SetGazeTarget(GazeTargetType.Camera);
    }
    
    /// <summary>
    /// Enable/disable micro-saccades
    /// </summary>
    public void SetMicroSaccadesEnabled(bool enabled)
    {
        enableMicroSaccades = enabled;
        if (!enabled)
        {
            saccadeOffset = Vector3.zero;
        }
    }
}

/// <summary>
/// Gaze target types (from PhaseDNA expression cues)
/// </summary>
public enum GazeTargetType
{
    Camera,    // Look at camera (default)
    Left,      // Look left
    Right,     // Look right
    Up,        // Look up
    Down,      // Look down
    Content    // Look at content area
}



