import streamlit as st
import torch
import torchvision
from PIL import Image
import numpy as np

# --- Model Loading and Preprocessing ---
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
resnet18.load_state_dict(torch.load("model/xray_classifier.pt"))
resnet18.to(device)  # Move the model to the GPU
resnet18.eval()

# Class names (adjust if yours are different)
class_names = ['covid', 'viral', 'normal']

# Image transformation
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
    with torch.no_grad():  # Disable gradient calculations
        output = resnet18(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name

# --- Streamlit App ---

st.title("X-ray Image Classifier")
st.write("This app classifies X-ray images into the following categories: COVID, Viral Pneumonia, Normal.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

# Show technical details or not based on checkbox.
show_details = st.sidebar.checkbox("Show Technical Details")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Make the prediction
    probabilities, predicted_class_index, predicted_class_name = predict_image_class(image)

    # Display the results
    st.write(f"**Prediction:** {predicted_class_name}")

    if show_details:
        st.write("**Probabilities:**")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_names[i]}: {prob:.4f}")
