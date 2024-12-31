# X-ray Image Classifier

This project is a Streamlit web application that classifies X-ray images into three categories: **COVID-19**, **Viral Pneumonia**, and **Normal**. It utilizes a pre-trained ResNet18 model, fine-tuned on a dataset of X-ray images, to perform the classification.

## Quick Demo

[Insert GIF or link to a short video demonstrating the app in action]

## Features

*   Upload X-ray images in common formats (PNG, JPG, JPEG).
*   Get fast predictions with probabilities for each class.
*   Option to display technical details (class probabilities).
*   GPU acceleration for enhanced performance (if a compatible GPU is available).
*   Easy deployment using Streamlit Sharing or other cloud platforms.

## Dependencies

This project relies on the following Python libraries:

*   **PyTorch:** Deep learning framework for model building and inference.
*   **Torchvision:** Datasets, transforms, and models for computer vision tasks.
*   **Pillow (PIL):** Image processing library.
*   **NumPy:** Numerical computing library for array operations.
*   **Streamlit:** Framework for building interactive web applications.

The specific versions used in the development environment are detailed in the `environment.yml` file.

## Installation

There are two main ways to set up this project:

**A. Using Conda (Recommended):**

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd xray-classifier-app
    ```

2.  **Create and activate the Conda environment:**

    *   **For CPU:**
        ```bash
        conda env create -f environment.yml
        conda activate xray_classifier
        ```
    *   **For GPU (if you have a compatible NVIDIA GPU and CUDA drivers):**
        *   Make sure your `environment.yml` file has the correct `pytorch` and `pytorch-cuda` (or `cudatoolkit`) dependencies that match your CUDA version (see PyTorch installation instructions for details: [https://pytorch.org/](https://pytorch.org/)).
        *   Then create and activate the environment:

            ```bash
            conda env create -f environment.yml
            conda activate xray_classifier
            ```

**B. Using Pip:**

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd xray-classifier-app
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate    # On Windows
    ```

3.  **Install dependencies using pip:**

    *   **For CPU:**
        ```bash
        pip install torch torchvision pillow numpy streamlit
        ```
    *   **For GPU:**
        *   Refer to the PyTorch website ([https://pytorch.org/](https://pytorch.org/)) for the correct `pip` installation command for your CUDA version. It might look like:
            ```bash
            pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
            pip install pillow numpy streamlit
            ```

## Usage

1.  **Start the Streamlit application:**

    ```bash
    streamlit run app/app.py
    ```

2.  Open your web browser and go to the URL provided in the terminal (usually `http://localhost:8501`).

3.  Upload an X-ray image using the file uploader.

4.  The application will display the image and the predicted class (COVID-19, Viral Pneumonia, or Normal).

5.  (Optional) Check the "Show Technical Details" box in the sidebar
