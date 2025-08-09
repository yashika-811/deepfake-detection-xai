import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Page configuration
st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🕵️‍♂️ Deepfake Detection App with Explainability")

st.markdown(
    """
    Upload an image to detect if it's **Real** or **Deepfake**.  
    You'll also see an **Explainability Heatmap** highlighting important regions.
    """
)

# ---------- Model Load Function ----------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ---------- Image Upload ----------
uploaded_file = st.file_uploader("Upload an Image (jpg/png)", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # ---------- Preprocessing ----------
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # ---------- Prediction ----------
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    # ---------- Label Mapping ----------
    label_map = {0: "✅ Real", 1: "⚠️ Deepfake"}
    label = label_map.get(pred_class, "Unknown")
    st.markdown(f"### Prediction: **{label}**")

    # ---------- Explainability (Grad-CAM) ----------
    with st.spinner("Generating Explainability Heatmap..."):
        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])
        grayscale_cam = grayscale_cam[0, :]

        # Convert PIL image to numpy
        img_np = np.array(image.resize((224,224))) / 255.0
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        st.subheader("🧭 Explainability Heatmap")
        st.image(visualization, caption='Model Attention', use_container_width=True)

st.markdown(
    """
    ---
    ✅ Created with ❤️ YASHI!!
    """
)


