using UnityEngine;
using Unity.Sentis;
using System.Collections;

public class BoneDetectorVideo : MonoBehaviour
{
    [Header("Modelo ONNX")]
    public ModelAsset modelAsset;
    public GameObject bone;

    [Header("Video")]
    public string videoPath = "video_hueso2.mp4";

    private Worker worker;
    private bool isProcessing = false;
    private UnityEngine.Video.VideoPlayer videoPlayer;
    private RenderTexture videoTexture;

    void Start()
    {
        var model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);
        Debug.Log("✅ Modelo ONNX cargado");

        videoTexture = new RenderTexture(640, 480, 0);

        videoPlayer = gameObject.AddComponent<UnityEngine.Video.VideoPlayer>();
        videoPlayer.url = System.IO.Path.Combine(Application.dataPath, videoPath);
        videoPlayer.targetTexture = videoTexture;
        videoPlayer.isLooping = true;
        videoPlayer.Play();

        Debug.Log($"✅ Vídeo iniciado: {videoPlayer.url}");
        StartCoroutine(DetectionLoop());
    }

    IEnumerator DetectionLoop()
    {
        yield return new WaitForSeconds(1f);

        while (true)
        {
            if (!isProcessing)
            {
                isProcessing = true;
                DetectBone();
                isProcessing = false;
            }
            yield return new WaitForSeconds(0.033f);
        }
    }

    void DetectBone()
    {
        var transform = new TextureTransform()
            .SetDimensions(640, 640, 3)
            .SetTensorLayout(TensorLayout.NCHW);

        var inputTensor = TextureConverter.ToTensor(videoTexture, transform);
        worker.Schedule(inputTensor);

        var output = worker.PeekOutput() as Tensor<float>;
        if (output != null)
        {
            var cpuOutput = output.ReadbackAndClone();
            ProcessDetections(cpuOutput);
            cpuOutput.Dispose();
        }

        inputTensor.Dispose();
    }

    void ProcessDetections(Tensor<float> output)
    {
        int numDetections = output.shape[1];
        float bestConf = 0f;
        float bestX = 0, bestY = 0, bestW = 0, bestH = 0;

        for (int i = 0; i < numDetections; i++)
        {
            float conf = output[0, i, 4];
            if (conf > bestConf && conf > 0.5f)
            {
                bestConf = conf;
                bestX = output[0, i, 0];
                bestY = output[0, i, 1];
                bestW = output[0, i, 2];
                bestH = output[0, i, 3];
            }
        }

        if (bestConf > 0.5f)
        {
            float unityX = (bestX / 640f - 0.5f) * 2f;
            float unityY = -(bestY / 640f - 0.5f) * 2f;
            float unityZ = 1f - (bestW * bestH) / (640f * 640f) * 5f;

            Vector3 targetPos = new Vector3(unityX, unityY, unityZ);

            bone.transform.position = Vector3.Lerp(
                bone.transform.position,
                targetPos,
                Time.deltaTime * 5f
            );

            Debug.Log($"🦴 Hueso detectado - Conf: {bestConf:F2} Pos: {targetPos}");
        }
        else
        {
            Debug.Log("👀 Sin detección en este frame");
        }
    }

    void OnDestroy()
    {
        worker?.Dispose();
        videoTexture?.Release();
    }
}