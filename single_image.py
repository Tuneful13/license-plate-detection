import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


def pipeline_matriculas():
    # --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
    print("1. Iniciando componentes...")
    

    model_path = "license_plate_detector.pt"
    
    # Cargar detectores
    detector = YOLO(model_path)
    reader = easyocr.Reader(['en'], gpu=True) # Cambia gpu=False si no tienes NVIDIA
    
    # --- 2. CARGAR IMAGEN ---
    # Usamos una URL de ejemplo
    url_imagen = 'data/image_demo.jpg'
    
    # Truco para leer imagen desde URL con OpenCV
    cap = cv2.VideoCapture(url_imagen)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: No se pudo cargar la imagen.")
        return

    # --- 3. DETECCIÓN (YOLO) ---
    print("2. Detectando ubicación de matrícula...")
    # conf=0.25 filtra detecciones débiles
    results = detector.predict(frame, save=False, conf=0.25, verbose=False)
    
    detection_found = False
    
    for result in results:
        for box in result.boxes.xyxy:
            detection_found = True
            
            # --- 4. RECORTE (CROP) ---
            x1, y1, x2, y2 = map(int, box)
            
            # Asegurar que las coordenadas estén dentro de la imagen
            h_img, w_img, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            # Extraer la región de interés (ROI)
            plate_roi = frame[y1:y2, x1:x2]
            
            # --- 5. LECTURA (EasyOCR) ---
            # detail=0 devuelve solo texto. allowlist filtra caracteres basura si quieres
            texto = reader.readtext(plate_roi, detail=0, paragraph=True)
            
            # Limpieza básica del texto
            texto_limpio = " ".join(texto).replace(" ", "").upper()
            
            print(f"--- RESULTADO ---")
            print(f"📍 Coordenadas: {x1},{y1} - {x2},{y2}")
            print(f"📖 Matrícula leída: {texto_limpio}")
            
            # (Opcional) Mostrar el recorte
            cv2.imshow("Placa", plate_roi)
            cv2.waitKey(0)

    if not detection_found:
        print("No se encontraron matrículas en la imagen.")

# Ejecutar
if __name__ == "__main__":
    pipeline_matriculas()