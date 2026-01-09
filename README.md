# Detección y Reconocimiento de Matrículas

Sistema de visión artificial para la detección y lectura de matrículas en tiempo real utilizando **YOLO** (Detección) y **EasyOCR** (Reconocimiento). Soporta procesamiento de imágenes estáticas, archivos de vídeo y flujo de cámara en directo.

---

## Características Principales
* **Tracking con YOLO:** Seguimiento persistente de vehículos mediante IDs únicos.
* **Lógica de Consenso (Buffer):** Filtra el ruido del OCR acumulando lecturas en un buffer para confirmar la matrícula por mayoría estadística.
* **Procesamiento Optimizado:** Redimensión inteligente para mantener altos FPS sin perder precisión en el OCR.

---

##  Instalación y Configuración

```bash
# Crear el entorno desde el archivo yml
conda env create -f environment.yml

conda activate vision_env
```

## Pruebas

1. Procesamiento de Imagen única

```python
python single_image.py
```

2. Procesamiento de Vídeo / Cámara en Vivo
Utiliza la lógica de tracking y buffer para una detección estable:

```python
python video_detection.py
```