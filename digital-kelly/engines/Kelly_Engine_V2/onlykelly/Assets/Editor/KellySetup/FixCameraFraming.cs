#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

namespace KellySetup
{
    /// <summary>
    /// Fixes camera framing for Kelly avatar
    /// Sets optimal position, rotation, and FOV for portrait view
    /// </summary>
    public class FixCameraFraming
    {
        [MenuItem("Kelly/📷 Fix Camera Framing")]
        public static void Fix()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                // Try to find any camera
                mainCamera = Object.FindObjectOfType<Camera>();
            }
            
            if (mainCamera == null)
            {
                Debug.LogError("❌ No camera found in scene! Please add a Main Camera.");
                return;
            }
            
            Debug.Log($"=== FIXING CAMERA: {mainCamera.name} ===");
            
            Undo.RecordObject(mainCamera.transform, "Fix Camera Framing");
            Undo.RecordObject(mainCamera, "Fix Camera Settings");
            
            // Set optimal position for portrait view (head to upper torso)
            mainCamera.transform.position = new Vector3(0, 1.5f, 2f);
            mainCamera.transform.rotation = Quaternion.Euler(0, 180f, 0);
            
            // Set FOV for proper framing
            mainCamera.fieldOfView = 40f;
            
            // Ensure proper clipping planes
            mainCamera.nearClipPlane = 0.1f;
            mainCamera.farClipPlane = 100f;
            
            Debug.Log($"✓ Camera Position: {mainCamera.transform.position}");
            Debug.Log($"✓ Camera Rotation: {mainCamera.transform.eulerAngles}");
            Debug.Log($"✓ Field of View: {mainCamera.fieldOfView}");
            Debug.Log($"✓ Clipping Planes: Near={mainCamera.nearClipPlane}, Far={mainCamera.farClipPlane}");
            Debug.Log("");
            Debug.Log("=== CAMERA FIX COMPLETE ===");
            Debug.Log("Check Game view to verify Kelly is properly framed!");
        }
        
        [MenuItem("Kelly/📷 Show Camera Info")]
        public static void ShowInfo()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                mainCamera = Object.FindObjectOfType<Camera>();
            }
            
            if (mainCamera == null)
            {
                Debug.LogError("❌ No camera found!");
                return;
            }
            
            Debug.Log("╔══════════════════════════════════════════════════════╗");
            Debug.Log("║              CAMERA INFORMATION                      ║");
            Debug.Log("╚══════════════════════════════════════════════════════╝");
            Debug.Log($"Camera: {mainCamera.name}");
            Debug.Log($"Position: {mainCamera.transform.position}");
            Debug.Log($"Rotation: {mainCamera.transform.eulerAngles}");
            Debug.Log($"Field of View: {mainCamera.fieldOfView}");
            Debug.Log($"Near Clip: {mainCamera.nearClipPlane}");
            Debug.Log($"Far Clip: {mainCamera.farClipPlane}");
            Debug.Log($"Projection: {(mainCamera.orthographic ? "Orthographic" : "Perspective")}");
            Debug.Log("");
            Debug.Log("RECOMMENDED VALUES:");
            Debug.Log("  Position: (0, 1.5, 2)");
            Debug.Log("  Rotation: (0, 180, 0)");
            Debug.Log("  FOV: 40");
        }
    }
}
#endif

