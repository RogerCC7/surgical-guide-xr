using UnityEngine;
using Unity.Sentis;
using System.Collections;

public class BoneDetector : MonoBehaviour
{
    [Header("Modelo ONNX")]
    public ModelAsset modelAsset;
    public GameObject femur3D;

    [Header("Camara compartida")]
    public static WebCamTexture sharedCamera;

    private Worker worker;
    private bool isProcessing = false;
    private bool cameraReady = false;

    void Start()
    {
        var model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);
        Debug.Log("✅ Modelo ONNX cargado con Sentis 2.x");

        sharedCamera = new WebCamTexture(640, 480, 30);
        sharedCamera.Play();
        Debug.Log("✅ Cámara iniciada");

        StartCoroutine(WaitForCamera());
    }

    IEnumerator WaitForCamera()
    {
        // Esperar hasta que la cámara tenga imagen real (no blanco)
        Debug.Log("⏳ Esperando imagen de cámara...");
        while (sharedCamera.width <= 16 || !sharedCamera.didUpdateThisFrame)
        {
            yield return new WaitForSeconds(0.1f);
        }
        // Espera extra para asegurar frames válidos
        yield return new WaitForSeconds(0.5f);
        cameraReady = true;
        Debug.Log($"✅ Cámara lista: {sharedCamera.width}x{sharedCamera.height}");
        StartCoroutine(DetectionLoop());
    }

    IEnumerator DetectionLoop()
    {
        while (true)
        {
            if (!isProcessing && sharedCamera != null && sharedCamera.didUpdateThisFrame && cameraReady)
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
        Texture2D frame = new Texture2D(sharedCamera.width, sharedCamera.height, TextureFormat.RGB24, false);
        frame.SetPixels(sharedCamera.GetPixels());
        frame.Apply();

        var textureTransform = new TextureTransform()
            .SetDimensions(640, 640, 3)
            .SetTensorLayout(TensorLayout.NCHW);

        var inputTensor = TextureConverter.ToTensor(frame, textureTransform);
        worker.Schedule(inputTensor);

        var output = worker.PeekOutput() as Tensor<float>;
        if (output != null)
        {
            var cpuOutput = output.ReadbackAndClone();
            ProcessDetections(cpuOutput);
            cpuOutput.Dispose();
        }

        inputTensor.Dispose();
        Destroy(frame);
    }

    void ProcessDetections(Tensor<float> output)
    {
        // Output shape Unity Sentis: (1, 37, 8400) o (1, 8400, 37)
        // Detectar el formato según shape
        int dim1 = output.shape[1];
        int dim2 = output.shape[2];

        float bestConf = 0f;
        float bestX = 0, bestY = 0, bestW = 0, bestH = 0;

        if (dim1 == 37 && dim2 == 8400)
        {
            // Formato (1, 37, 8400): fila 4 = confianza
            for (int i = 0; i < 8400; i++)
            {
                float conf = output[0, 4, i];
                if (conf > bestConf)
                {
                    bestConf = conf;
                    bestX = output[0, 0, i];
                    bestY = output[0, 1, i];
                    bestW = output[0, 2, i];
                    bestH = output[0, 3, i];
                }
            }
        }
        else if (dim1 == 8400 && dim2 == 37)
        {
            // Formato (1, 8400, 37): columna 4 = confianza
            for (int i = 0; i < 8400; i++)
            {
                float conf = output[0, i, 4];
                if (conf > bestConf)
                {
                    bestConf = conf;
                    bestX = output[0, i, 0];
                    bestY = output[0, i, 1];
                    bestW = output[0, i, 2];
                    bestH = output[0, i, 3];
                }
            }
        }
        else
        {
            Debug.LogWarning($"Shape inesperado: {output.shape}");
            return;
        }

        // Umbral bajo porque los valores son raw sin sigmoid
        if (bestConf > 0.5f)
        {
            float screenX = (bestX / 640f) * Screen.width;
            float screenY = (1f - bestY / 640f) * Screen.height;
            Vector3 screenPos = new Vector3(screenX, screenY, 50f);
            Vector3 worldPos = Camera.main.ScreenToWorldPoint(screenPos);

            if (femur3D != null)
            {
                femur3D.transform.position = Vector3.Lerp(
                    femur3D.transform.position,
                    worldPos,
                    Time.deltaTime * 5f
                );
            }

            Debug.Log($"🦴 Hueso detectado - Conf: {bestConf:F2}");
        }
    }

    void OnDestroy()
    {
        worker?.Dispose();
        sharedCamera?.Stop();
    }
}