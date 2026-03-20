import cv2
import numpy as np
import onnxruntime as ort

yolo_session = ort.InferenceSession('best.onnx')
frame = cv2.imread('referencias/hueso_tumor.jpg')

img = cv2.resize(frame, (640, 640))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = img_rgb.astype(np.float32) / 255.0
img_tensor = np.transpose(img_tensor, (2, 0, 1))
img_tensor = np.expand_dims(img_tensor, 0)

input_name = yolo_session.get_inputs()[0].name
outputs = yolo_session.run(None, {input_name: img_tensor})
output = outputs[0][0]

print(f"Shape output: {output.shape}")
max_conf = 0
for det in output.T:
    conf = det[4] if len(det) > 4 else 0
    if conf > max_conf:
        max_conf = conf

print(f"Confianza máxima detectada: {max_conf}")