using UnityEngine;

public class AutoBlink : MonoBehaviour
{
    public SkinnedMeshRenderer smr;
    public string left = "eyeBlink_L";
    public string right = "eyeBlink_R";
    public float speed = 30f;
    public float hold = 0.06f;

    private int li = -1;
    private int ri = -1;
    private float next;
    private float val;
    private bool closing;
    private float holdUntil;

    void Start()
    {
        var m = smr.sharedMesh;
        for (int i = 0; i < m.blendShapeCount; i++)
        {
            var n = m.GetBlendShapeName(i).ToLower();
            if (n.Contains("eyeblink_l"))
                li = i;
            if (n.Contains("eyeblink_r"))
                ri = i;
        }
        Schedule();
    }

    void Schedule()
    {
        next = Time.time + Random.Range(3f, 6f);
        closing = true;
        val = 0;
    }

    void Update()
    {
        if (Time.time < next)
            return;

        float step = speed * Time.deltaTime;

        if (closing)
        {
            val += step;
            if (val >= 100f)
            {
                val = 100f;
                closing = false;
                holdUntil = Time.time + hold;
            }
        }
        else if (Time.time > holdUntil)
        {
            val -= step;
            if (val <= 0)
            {
                val = 0;
                Schedule();
            }
        }

        if (li >= 0)
            smr.SetBlendShapeWeight(li, val);
        if (ri >= 0)
            smr.SetBlendShapeWeight(ri, val);
    }
}








