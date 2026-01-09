import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import os
from collections import Counter # Para contar los votos del buffer

# --- 1. INICIALIZACIÓN GLOBAL ---
print("Cargando modelos...")
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'], gpu=True) 

def pipeline_final(vid_path):
    cap = cv2.VideoCapture(vid_path)
    
    # --- CONFIGURACIÓN DEL BUFFER ---
    target_width = 640
    ocr_interval = 2        
    tamano_buffer = 12      # Guardamos las últimas 12 lecturas
    votos_necesarios = 5    # Si una matrícula aparece 5 veces en el buffer, se confirma
    
    # Estructura: {id_seguimiento: {'buffer': [], 'confirmado': False, 'resultado': ""}}
    memoria_votos = {} 
    lista_oficial = [] 

    print(f"Modo Buffer Activo. Umbral: {votos_necesarios} votos de {tamano_buffer} posibles.")

    while cap.isOpened():
        ret, frame_orig = cap.read()
        if not ret: break

        h_orig, w_orig = frame_orig.shape[:2]
        ratio = target_width / float(w_orig) if w_orig > target_width else 1.0
        frame_resized = cv2.resize(frame_orig, (target_width, int(h_orig * ratio)))

        results = model.track(frame_resized, persist=True, conf=0.5, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = [int(coord / ratio) for coord in box]

                # Inicializar ID si es nuevo
                if obj_id not in memoria_votos:
                    memoria_votos[obj_id] = {'buffer': [], 'confirmado': False, 'resultado': ""}

                # --- PROCESO DE VOTO ---
                if not memoria_votos[obj_id]['confirmado'] and frame_nmr % ocr_interval == 0:
                    plate_roi = frame_orig[max(0, y1):y2, max(0, x1):x2]
                    if plate_roi.size > 0:
                        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                        res = reader.readtext(gray, detail=0)
                        
                        if res:
                            texto_leido = "".join(res).upper().replace(" ", "")
                            # Añadir al buffer
                            memoria_votos[obj_id]['buffer'].append(texto_leido)
                            
                            # Mantener el tamaño del buffer
                            if len(memoria_votos[obj_id]['buffer']) > tamano_buffer:
                                memoria_votos[obj_id]['buffer'].pop(0)

                            # CONTAR VOTOS
                            conteo = Counter(memoria_votos[obj_id]['buffer'])
                            mas_comun, num_votos = conteo.most_common(1)[0]

                            # ¿Tenemos un ganador sólido?
                            if num_votos >= votos_necesarios:
                                memoria_votos[obj_id]['confirmado'] = True
                                memoria_votos[obj_id]['resultado'] = mas_comun
                                if mas_comun not in lista_oficial:
                                    lista_oficial.append(mas_comun)
                                    print(f"\n[✔] CONFIRMADO POR MAYORÍA (ID {obj_id}): {mas_comun} ({num_votos}/{len(memoria_votos[obj_id]['buffer'])} votos)")

                # --- VISUALIZACIÓN ---
                datos = memoria_votos[obj_id]
                if datos['confirmado']:
                    label = f"OK: {datos['resultado']}"
                    color = (0, 255, 0)
                else:
                    # Mostrar la tendencia actual mientras decide
                    current_len = len(datos['buffer'])
                    label = f"Procesando... ({current_len}/{tamano_buffer})"
                    color = (0, 165, 255)

                cv2.rectangle(frame_orig, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_orig, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Visualización redimensionada
        frame_display = cv2.resize(frame_orig, (1280, int(h_orig * (1280/w_orig))))
        cv2.imshow("LPR Buffer Mode", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("\nMATRÍCULAS FINALES:", lista_oficial)

if __name__ == "__main__":
    frame_nmr = 0 # Variable global de apoyo
    ruta = "data/demo_vid.mp4"
    if os.path.exists(ruta): pipeline_final(ruta)