using UnityEngine;
using System.IO;
using System.Collections;

public class DatasetCapture : MonoBehaviour
{
    [Header("Configuración")]
    public GameObject bone;
    public Camera captureCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;

    [Header("Ángulos")]
    public int anglesH = 36;
    public int anglesV = 5;
    public float[] distances = { 0.3f, 0.5f, 0.7f };

    [Header("Output")]
    public string outputPath = "Assets/Dataset/";

    void Start()
    {
        Directory.CreateDirectory(outputPath);
        StartCoroutine(CaptureAll());
    }

    IEnumerator CaptureAll()
    {
        int index = 0;
        string poseLog = "filename,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z\n";

        foreach (float dist in distances)
        {
            for (int v = 0; v < anglesV; v++)
            {
                float pitch = Mathf.Lerp(-30f, 30f, v / (float)(anglesV - 1));

                for (int h = 0; h < anglesH; h++)
                {
                    float yaw = h * (360f / anglesH);

                    Quaternion rot = Quaternion.Euler(pitch, yaw, 0);
                    Vector3 pos = bone.transform.position + rot * Vector3.forward * -dist;
                    captureCamera.transform.position = pos;
                    captureCamera.transform.LookAt(bone.transform.position);

                    yield return new WaitForEndOfFrame();

                    string filename = $"bone_{index:D4}.png";
                    Capture(filename);

                    Vector3 camPos = captureCamera.transform.position;
                    Vector3 camRot = captureCamera.transform.eulerAngles;
                    poseLog += $"{filename},{camPos.x:F4},{camPos.y:F4},{camPos.z:F4}," +
                               $"{camRot.x:F4},{camRot.y:F4},{camRot.z:F4}\n";

                    index++;
                    Debug.Log($"Capturada imagen {index} de 540");
                }
            }
        }

        File.WriteAllText(outputPath + "poses.csv", poseLog);
        Debug.Log($"✅ DATASET COMPLETO: {index} imágenes guardadas en {outputPath}");
    }

    void Capture(string filename)
    {
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        captureCamera.targetTexture = rt;
        captureCamera.Render();

        RenderTexture.active = rt;
        Texture2D img = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        img.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        img.Apply();

        File.WriteAllBytes(outputPath + filename, img.EncodeToPNG());

        captureCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
    }
}