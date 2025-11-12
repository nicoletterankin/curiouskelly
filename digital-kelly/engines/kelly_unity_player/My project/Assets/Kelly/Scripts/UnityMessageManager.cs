using System;
using UnityEngine;

/// <summary>
/// Manages bidirectional communication between Unity and Flutter
/// </summary>
public class UnityMessageManager : MonoBehaviour
{
    public static UnityMessageManager Instance { get; private set; }
    
    public event Action<string> OnMessage;
    
    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    /// <summary>
    /// Called by Flutter to send messages to Unity
    /// </summary>
    public void ReceiveMessageFromFlutter(string message)
    {
        Debug.Log($"[UnityMessageManager] Received from Flutter: {message}");
        OnMessage?.Invoke(message);
    }

    /// <summary>
    /// Send message from Unity to Flutter
    /// </summary>
    public void SendMessageToFlutter(string message)
    {
        Debug.Log($"[UnityMessageManager] Sending to Flutter: {message}");
        
#if UNITY_ANDROID && !UNITY_EDITOR
        // Android implementation
        using (AndroidJavaClass unityClass = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
        {
            using (AndroidJavaObject currentActivity = unityClass.GetStatic<AndroidJavaObject>("currentActivity"))
            {
                currentActivity.Call("receiveUnityMessage", message);
            }
        }
#elif UNITY_IOS && !UNITY_EDITOR
        // iOS implementation
        SendMessageToFlutter_iOS(message);
#else
        // Editor simulation
        Debug.Log($"[SIMULATED] Flutter would receive: {message}");
#endif
    }

#if UNITY_IOS && !UNITY_EDITOR
    [System.Runtime.InteropServices.DllImport("__Internal")]
    private static extern void SendMessageToFlutter_iOS(string message);
#endif
}




