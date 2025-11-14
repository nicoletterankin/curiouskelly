using UnityEngine;

public class BreathingLayer : MonoBehaviour
{
    public SkinnedMeshRenderer smr;
    public string shapeName = "breathMicro";
    public float amp = 4f;
    public float freq = 0.25f;
    private int idx = -1;

    void Start()
    {
        var m = smr.sharedMesh;
        for (int i = 0; i < m.blendShapeCount; i++)
        {
            if (m.GetBlendShapeName(i).ToLower().Contains(shapeName.ToLower()))
            {
                idx = i;
                break;
            }
        }
    }

    void Update()
    {
        if (idx < 0)
            return;

        float w = (Mathf.Sin(Time.time * 2 * Mathf.PI * freq) * 0.5f + 0.5f) * amp;
        smr.SetBlendShapeWeight(idx, w);
    }
}



















