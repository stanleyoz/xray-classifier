import gradio as gr
import torch
import torchvision
from PIL import Image
import numpy as np

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

# --- Gradio Interface ---
def classify_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    probabilities, predicted_class_index, predicted_class_name = predict_image_class(image)
    return predicted_class_name, {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

# Use gr.Image, gr.Textbox, and gr.Label directly
inputs = gr.Image(type="numpy")
outputs = [gr.Textbox(label="Prediction"), gr.Label(num_top_classes=3, label="Probabilities")]

title = "X-ray Image Classifier"
description = "This app classifies X-ray images into the following categories: COVID, Viral Pneumonia, Normal."

iface = gr.Interface(
    fn=classify_image,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    #examples=[["examples/example1.jpg"]],  # Uncomment if you have examples
)

iface.launch(share=True)
#iface.launch(share=False)
