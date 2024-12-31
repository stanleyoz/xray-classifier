import streamlit as st
import torch
import torchvision
from PIL import Image
import numpy as np
import argparse  # Import the argparse module

# --- Model Loading and Preprocessing ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
resnet18.load_state_dict(torch.load("model/xray_classifier.pt", map_location=device))
resnet18.to(device)
resnet18.eval()

class_names = ['covid', 'viral', 'normal']

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Prediction Function ---
def predict_image_class(image):
    image = image.convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = resnet18(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name

# --- Streamlit App ---
st.title("X-ray Image Classifier")
st.write("This app classifies X-ray images into the following categories: COVID, Viral Pneumonia, Normal.")

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("-public", action="store_true", help="Enable public access via Streamlit Sharing")
args = parser.parse_args()

# Check if the app is running on Streamlit Sharing (public cloud)
if args.public:
    # --- Public Access Configuration ---
    # (You might not need any specific code here for Streamlit Sharing)
    st.sidebar.info("Running in public mode (Streamlit Sharing).") 
else:
    # --- Local Development Mode ---
    st.sidebar.info("Running in local mode.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

show_details = st.sidebar.checkbox("Show Technical Details")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    probabilities, predicted_class_index, predicted_class_name = predict_image_class(image)

    st.write(f"**Prediction:** {predicted_class_name}")

    if show_details:
        st.write("**Probabilities:**")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_names[i]}: {prob:.4f}")
