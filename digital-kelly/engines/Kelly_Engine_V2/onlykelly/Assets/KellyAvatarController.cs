using UnityEngine;
using System.Collections.Generic;

public class KellyAvatarController : MonoBehaviour
{
    [Header("Face Configuration")]
    public SkinnedMeshRenderer faceMesh; // Drag her face here
    public int visemeMultiplier = 100;

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