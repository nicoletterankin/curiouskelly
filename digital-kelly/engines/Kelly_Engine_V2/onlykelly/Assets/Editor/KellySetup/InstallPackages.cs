#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;

namespace KellySetup
{
    public class InstallPackages : EditorWindow
    {
        private static AddRequest[] requests;
        
        [MenuItem("Kelly/2. Install Required Packages")]
        public static void Install()
        {
            Debug.Log("=== INSTALLING REQUIRED PACKAGES ===");
            
            string[] packages = new string[]
            {
                "com.unity.burst",
                "com.unity.mathematics",
                "com.unity.collections",
                "com.unity.jobs",
                "com.unity.addressables",
                "com.unity.timeline",
                "com.unity.cinemachine"
            };
            
            requests = new AddRequest[packages.Length];
            for (int i = 0; i < packages.Length; i++)
            {
                requests[i] = Client.Add(packages[i]);
                Debug.Log($"Installing: {packages[i]}");
            }
            
            EditorApplication.update += Progress;
        }
        
        private static void Progress()
        {
            bool allDone = true;
            foreach (var req in requests)
            {
                if (!req.IsCompleted) allDone = false;
                if (req.Status == StatusCode.Success)
                    Debug.Log($"✓ Installed: {req.Result.displayName}");
                else if (req.Status >= StatusCode.Failure)
                    Debug.LogError($"✗ Failed: {req.Error.message}");
            }
            
            if (allDone)
            {
                EditorApplication.update -= Progress;
                Debug.Log("=== PACKAGE INSTALLATION COMPLETE ===");
            }
        }
    }
}
#endif

