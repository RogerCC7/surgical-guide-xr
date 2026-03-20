import cv2
import torch
import numpy as np
import socket
import json
import struct
import onnxruntime as ort
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Usando: {device}")

base_path = os.path.dirname(os.path.abspath(__file__))

extractor = SuperPoint(max_num_keypoints=512).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

yolo_session = ort.InferenceSession(os.path.join(base_path, 'best.onnx'))

ref_tumor = load_image(os.path.join(base_path, 'referencias', 'hueso_tumor.jpg')).to(device)
ref_donante = load_image(os.path.join(base_path, 'referencias', 'hueso_donante.jpg')).to(device)

with torch.no_grad():
    feats_tumor = extractor.extract(ref_tumor)
    feats_donante = extractor.extract(ref_donante)

print('✅ Servidor listo en 0.0.0.0:65432. Esperando conexión...')

def run_yolo_onnx(frame):
    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, 0)

    input_name = yolo_session.get_inputs()[0].name
    outputs = yolo_session.run(None, {input_name: img_tensor})
    output = outputs[0][0]  # (37, 8400)

    boxes = output[:4, :]
    confs = output[4, :]

    best_idx = np.argmax(confs)
    best_conf = confs[best_idx]

    if best_conf < 0.5:
        return None, 0

    h, w = frame.shape[:2]
    cx, cy, bw, bh = boxes[:, best_idx]
    x1 = int((cx - bw/2) / 640 * w)
    y1 = int((cy - bh/2) / 640 * h)
    x2 = int((cx + bw/2) / 640 * w)
    y2 = int((cy + bh/2) / 640 * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return [x1, y1, x2, y2], float(best_conf)

def recv_exact(conn, n):
    """Recibir exactamente n bytes."""
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Conexión cerrada")
        data += chunk
    return data

def handle_client(conn, addr):
    print(f"📡 Conexión persistente desde: {addr}")
    try:
        while True:
            # Leer 4 bytes del tamaño
            size_data = recv_exact(conn, 4)
            img_size = struct.unpack('<I', size_data)[0]

            if img_size == 0 or img_size > 10_000_000:
                print(f"Tamaño inválido: {img_size}")
                break

            # Leer imagen completa
            img_data = recv_exact(conn, img_size)

            img_array = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                response = json.dumps({
                    'detected': 'none', 'tumor_matches': 0,
                    'donante_matches': 0, 'bbox_x1': 0,
                    'bbox_y1': 0, 'bbox_x2': 0, 'bbox_y2': 0
                }).encode()
                conn.sendall(response)
                continue

            # YOLO
            bbox, conf = run_yolo_onnx(frame)

            if bbox is None:
                response = json.dumps({
                    'detected': 'none', 'tumor_matches': 0,
                    'donante_matches': 0, 'bbox_x1': 0,
                    'bbox_y1': 0, 'bbox_x2': 0, 'bbox_y2': 0
                }).encode()
                conn.sendall(response)
                continue

            # ROI
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                response = json.dumps({
                    'detected': 'none', 'tumor_matches': 0,
                    'donante_matches': 0, 'bbox_x1': 0,
                    'bbox_y1': 0, 'bbox_x2': 0, 'bbox_y2': 0
                }).encode()
                conn.sendall(response)
                continue

            # LightGlue
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_tensor = torch.tensor(roi_rgb).permute(2, 0, 1).float() / 255.0
            roi_tensor = roi_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                feats_roi = extractor.extract(roi_tensor)
                matches_t = matcher({'image0': feats_tumor, 'image1': feats_roi})
                n_tumor = len(rbd(matches_t)['matches'])
                matches_d = matcher({'image0': feats_donante, 'image1': feats_roi})
                n_donante = len(rbd(matches_d)['matches'])

            detected = "none"
            if n_tumor > n_donante and n_tumor > 5:
                detected = "tumor"
            elif n_donante > n_tumor and n_donante > 5:
                detected = "donante"

            result = {
                'tumor_matches': n_tumor,
                'donante_matches': n_donante,
                'detected': detected,
                'bbox_x1': float(bbox[0]),
                'bbox_y1': float(bbox[1]),
                'bbox_x2': float(bbox[2]),
                'bbox_y2': float(bbox[3]),
                'conf': float(conf)
            }
            response = json.dumps(result).encode()
            conn.sendall(response)
            print(f"YOLO({conf:.2f}) -> ROI -> LightGlue: T:{n_tumor} D:{n_donante} -> {detected}")

    except Exception as e:
        print(f"❌ Cliente desconectado {addr}: {e}")
    finally:
        conn.close()
        print(f"🔌 Conexión cerrada: {addr}")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 65432))
server.listen(5)

import threading

while True:
    conn, addr = server.accept()
    thread = threading.Thread(target=handle_client, args=(conn, addr))
    thread.daemon = True
    thread.start()