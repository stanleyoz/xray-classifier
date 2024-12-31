# X-ray Image Classifier

This project provides two implementations of a web application for classifying X-ray images into three categories: **COVID-19**, **Viral Pneumonia**, and **Normal**. The application utilizes a pre-trained ResNet18 model, fine-tuned on a dataset of X-ray images.

## Implementations

This repository contains two versions of the X-ray classifier app:

1.  **Streamlit Version (`app_streamlit.py`):** A feature-rich web application built using the Streamlit framework, offering a more interactive and visually appealing user interface.
2.  **Gradio Version (`app_gradio.py`):** A simpler, quickly deployable version using the Gradio library, ideal for rapid prototyping and sharing demos via temporary public links.

## Dependencies

Both versions of the app rely on the following Python libraries:

*   **PyTorch:** Deep learning framework for model building and inference.
*   **Torchvision:** Datasets, transforms, and models for computer vision tasks.
*   **Pillow (PIL):** Image processing library.
*   **NumPy:** Numerical computing library for array operations.
*   **Streamlit:** Framework for building interactive web applications (Streamlit version only).
*   **Gradio:** Framework for creating quick and easy web UIs for machine learning models (Gradio version only).

The specific versions used in the development environment are detailed in the `environment.yml` file.

## Installation

It is recommended to use a Conda environment to manage the dependencies:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/stanleyoz/xray-classifier
    cd xray-classifier
    ```

2.  **Create and activate the Conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate xray-classifier
    ```

## Usage

### Streamlit Version (`app_streamlit.py`)

1.  **Navigate to the project directory:**

    ```bash
    cd <path/to/your/repo>/xray-classifier
    ```

2.  **Run the Streamlit app:**

    ```bash
    streamlit run app_streamlit.py
    ```

3.  Open your web browser and go to the URL provided in the terminal (usually `http://localhost:8501`).

4.  Upload an X-ray image and get the classification results.

### Gradio Version (`app_gradio.py`)

1.  **Navigate to the project directory:**

    ```bash
    cd <path/to/your/repo>/xray-classifier
    ```

2.  **Run the Gradio app:**

    ```bash
    python app_gradio.py
    ```

3.  Open your web browser and go to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

4.  **For a temporary public link:**
    *   The Gradio version can be launched with a temporary public URL using:

        ```bash
        python app_gradio.py
        ```

        This will provide a `gradio.live` link that is active for 72 hours.

### Switching Between Versions

This repository uses Git branches to manage the different versions:

*   `main` branch: Contains the core code, including both Streamlit and Gradio versions
*   `gradio-version` branch: (Optional) You can use this branch to experiment with or develop the Gradio version separately.

To switch between branches, use the following Git commands:

*   Switch to `main` branch: `git checkout main`

## Model Details

The classification model is a ResNet18 architecture pre-trained on the ImageNet dataset. It has been fine-tuned on a dataset of X-ray images to specialize in COVID-19, Viral Pneumonia, and Normal classification. The model weights are saved in the `model/xray_classifier.pt` file.

## Contributing

Contributions are welcome! If you want to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch: `git checkout -b feature/your-feature-name`
3.  Make your changes and commit them: `git commit -m "Add: your feature description"`
4.  Push to the branch: `git push origin feature/your-feature-name`
5.  Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for informational and educational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
