using System.Collections;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class LightGlueClient : MonoBehaviour
{
    public string pc_ip_address = "192.168.1.161";
    public int port = 65432;
    public GameObject hueso_tumor;
    public GameObject hueso_donante;
    public Camera ar_camera;
    public RawImage camera_display;

    private TcpClient client;
    private NetworkStream stream;
    private bool isConnected = false;
    private float sendInterval = 0.5f;
    private float timer = 0f;

    void Start()
    {
        StartCoroutine(WaitForCameraAndStart());
    }

    IEnumerator WaitForCameraAndStart()
    {
        while (BoneDetector.sharedCamera == null || !BoneDetector.sharedCamera.isPlaying)
        {
            yield return new WaitForSeconds(0.1f);
        }
        Debug.Log("✅ LightGlueClient: camara compartida lista");
        StartCoroutine(ConnectLoop());
    }

    IEnumerator ConnectLoop()
    {
        while (true)
        {
            if (!isConnected)
            {
                yield return StartCoroutine(TryConnect());
            }
            yield return new WaitForSeconds(0.5f);
        }
    }

    IEnumerator TryConnect()
    {
        Debug.Log($"Intentando conectar a {pc_ip_address}:{port}...");
        bool success = false;
        string errorMsg = "";

        System.Threading.Thread connectThread = new System.Threading.Thread(() =>
        {
            try
            {
                TcpClient newClient = new TcpClient();
                newClient.ReceiveTimeout = 15000;
                newClient.SendTimeout = 15000;
                newClient.Connect(pc_ip_address, port);
                client = newClient;
                stream = client.GetStream();
                success = true;
            }
            catch (System.Exception e)
            {
                errorMsg = e.Message;
                success = false;
            }
        });

        connectThread.Start();

        while (connectThread.IsAlive)
        {
            yield return null;
        }

        if (success)
        {
            isConnected = true;
            Debug.Log("✅ Conectado al servidor (conexión persistente)");
        }
        else
        {
            isConnected = false;
            Debug.LogWarning($"❌ No se pudo conectar: {errorMsg}");
            yield return new WaitForSeconds(2f);
        }
    }

    void Update()
    {
        if (!isConnected) return;

        timer += Time.deltaTime;
        if (timer >= sendInterval)
        {
            timer = 0f;
            SendFrameToLightGlue();
        }
    }

    void SendFrameToLightGlue()
    {
        if (BoneDetector.sharedCamera == null || !BoneDetector.sharedCamera.isPlaying) return;
        StartCoroutine(ProcessFrame());
    }

    IEnumerator ProcessFrame()
    {
        yield return new WaitForEndOfFrame();

        WebCamTexture cam = BoneDetector.sharedCamera;
        Texture2D tex = new Texture2D(cam.width, cam.height, TextureFormat.RGB24, false);
        tex.SetPixels(cam.GetPixels());
        tex.Apply();

        if (camera_display != null)
            camera_display.texture = cam;

        byte[] jpgBytes = tex.EncodeToJPG(75);
        Destroy(tex);

        string result = SendToServer(jpgBytes);
        if (result != null)
        {
            ProcessResult(result);
        }
    }

    string SendToServer(byte[] imageBytes)
    {
        if (!isConnected || stream == null) return null;

        try
        {
            byte[] sizeBytes = System.BitConverter.GetBytes(imageBytes.Length);
            stream.Write(sizeBytes, 0, 4);
            stream.Write(imageBytes, 0, imageBytes.Length);

            byte[] buffer = new byte[4096];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            if (bytesRead > 0)
            {
                return Encoding.UTF8.GetString(buffer, 0, bytesRead);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"❌ Error en comunicación: {e.Message}. Reconectando...");
            isConnected = false;
            try { client?.Close(); } catch { }
        }
        return null;
    }

    void ProcessResult(string jsonString)
    {
        try
        {
            var result = JsonUtility.FromJson<DetectionResult>(jsonString);

            Debug.Log($"T:{result.tumor_matches} D:{result.donante_matches} -> {result.detected}");

            if (result.detected == "tumor" && hueso_tumor != null)
            {
                MoveObjectToBbox(hueso_tumor, result);
                hueso_tumor.SetActive(true);
                if (hueso_donante != null) hueso_donante.SetActive(false);
            }
            else if (result.detected == "donante" && hueso_donante != null)
            {
                MoveObjectToBbox(hueso_donante, result);
                hueso_donante.SetActive(true);
                if (hueso_tumor != null) hueso_tumor.SetActive(false);
            }
            // Si es "none" no hacemos nada, el objeto se queda donde estaba
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Error parseando JSON: {e.Message}");
        }
    }

    void MoveObjectToBbox(GameObject obj, DetectionResult result)
    {
        WebCamTexture cam = BoneDetector.sharedCamera;
        if (cam == null || ar_camera == null) return;

        // Centro del bbox en píxeles de la imagen original (1920x1080)
        float bboxCenterX = (result.bbox_x1 + result.bbox_x2) / 2f;
        float bboxCenterY = (result.bbox_y1 + result.bbox_y2) / 2f;

        // Normalizar entre 0 y 1
        // cam.width = 1920, cam.height = 1080
        float normX = bboxCenterX / cam.width;
        // Y invertida: en imagen 0 es arriba, en Unity 0 es abajo
        float normY = 1f - (bboxCenterY / cam.height);

        // Convertir a coordenadas de pantalla en píxeles
        float screenX = normX * Screen.width;
        float screenY = normY * Screen.height;

        // Profundidad: 2 metros delante de la cámara
        // Puedes cambiar este valor si el fémur aparece muy cerca o muy lejos
        float depth = 2f;

        Vector3 screenPos = new Vector3(screenX, screenY, depth);
        Vector3 worldPos = ar_camera.ScreenToWorldPoint(screenPos);

        // Lerp suave para que no salte bruscamente
        // El 8f controla la velocidad: más alto = más rápido
        obj.transform.position = Vector3.Lerp(
            obj.transform.position,
            worldPos,
            Time.deltaTime * 8f
        );
    }

    void OnDestroy()
    {
        try { stream?.Close(); } catch { }
        try { client?.Close(); } catch { }
    }

    [System.Serializable]
    class DetectionResult
    {
        public string detected;
        public int tumor_matches;
        public int donante_matches;
        public float bbox_x1;
        public float bbox_y1;
        public float bbox_x2;
        public float bbox_y2;
        public float conf;
    }
}