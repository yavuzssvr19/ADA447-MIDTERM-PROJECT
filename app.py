import gradio as gr
from fastai.vision.all import *
import torch
import pathlib
import sys

# Windows için pathlib ayarı
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# Modeli yükle
try:
    learn = load_learner("pneumonia_resnet34_transfer.pkl", cpu=True)
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    # Alternatif yükleme yöntemi
    state = torch.load("pneumonia_resnet34_transfer.pkl", map_location='cpu')
    learn = Learner.load("pneumonia_resnet34_transfer.pkl", state)

# Tahmin fonksiyonu
def predict(img):
    pred_class, pred_idx, probs = learn.predict(img)
    return {
        "NORMAL": float(probs[0]),
        "PNEUMONIA": float(probs[1])
    }

# Arayüz tanımı
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia Detection from Chest X-ray",
    description="Upload a chest X-ray image. The model will predict if it shows signs of pneumonia or is normal."
)

# Çalıştır
interface.launch()
