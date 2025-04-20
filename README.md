# Smart Rice Leaf Disease Detection: Revolutionizing Paddy Crop Health with AI

This project aims to detect various diseases on rice leaves using **Deep Learning**. By utilizing a dataset of rice leaf images affected by different diseases, this model can help farmers automatically identify diseases, facilitate plant care, and improve agricultural productivity.

## Project Features

- **Rice Leaf Disease Detection**: Using a deep learning model to classify rice leaf diseases based on images.
- **Implemented Models**: Utilizes **Convolutional Neural Networks (CNN)** for image classification.
- **Integrated Models**:
  - **TensorFlow**: The deep learning model is stored in the TensorFlow format.
  - **TFLite**: The model is available in **TFLite** format for mobile applications.
  - **TensorFlow.js**: The model is also available for web-based applications using TensorFlow.js.

## Model Architecture

The model architecture used for classifying rice leaf diseases is based on **Convolutional Neural Networks (CNN)** with **VGG16** as the base feature extractor. Below are the key layers and configuration:

- **Base Model: VGG16 pre-trained model as a feature extractor**
- **Custom Layers**:
  - **Conv2D Layer**: 128 filters with kernel size (3, 3), ReLU activation.
  - **MaxPooling2D Layer**: Pooling size of (2, 2).
  - **Flatten Layer**: To reshape the output for the fully connected layers.
  - **Dense Layer**: 128 units with ReLU activation.
  - **Dropout Layer**: Dropout rate of 0.4 to prevent overfitting.
  - **Output Layer**: Dense layer with 9 units (for 9 classes), using Softmax activation for classification.
- **Loss Function: Categorical Crossentropy**
- **Optimizer: Adam with learning rate 0.0005**
- **Callbacks**:
  - **EarlyStopping**
  - **ReduceLROnPlateau**
  - **ModelCheckpoint**

## Model Training, Evaluation & Deployment
The rice leaf disease detection model was trained using a CNN architecture with a VGG16 base and custom layers. The training process was carefully monitored using callbacks such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint to optimize performance and prevent overfitting.

 **Training Results**
  - **Training Accuracy: 99.70%**
  - **Training Loss: 0.0086**
  - **Validation Accuracy: 96.52%**
  - **Validation Loss: 0.3192 (by the 24th epoch)**

**Testing Results**
- **Test Accuracy: 95.16%**
- **Test Loss: 0.2228**

**Model Export**
- TensorFlow SavedModel – for standard deployment and reuse.
- TensorFlow Lite (TFLite) – optimized for mobile devices, enabling on-device inference in the field.
- TensorFlow.js (TFJS) – for web-based deployment, allowing farmers or users to run disease detection directly in the browser.

## Future Work
- **Improving Model Accuracy**: Despite achieving excellent results, there’s always room for further enhancement. Techniques like **data augmentation**, **fine-tuning**, and experimenting with **deeper architectures** could help improve accuracy even more.
- **Real-time Deployment**: The TFLite and TFJS models are ready for real-time deployment, making it possible to detect rice leaf diseases in the field using mobile devices or web applications.
- **Integration with Agricultural Systems**: This model could be integrated into agricultural tools or mobile apps, providing farmers with real-time diagnostics, helping them manage crop health, and preventing the spread of diseases.

## Dataset
The dataset used in this project contains images of rice leaves with the following diseases and a healthy class:
- **Bacterial Leaf Blight**
- **Brown Spot**
- **Healthy Rice Leaf**
- **Leaf Blast**
- **Leaf Scald**
- **Narrow Brown Leaf Spot**
- **Rice Hispa**
- **Sheath Blight**

## Project Structure
```sh
Submission/
├── saved_model/
│   ├── fingerprint.pb
│   ├── saved_model.pb  
│   └── variables/  
│       ├── variables.index
│       └── variables.date-00000-of-00001 
├── tfjs_model/
│   ├── group1-shard1of15.bin                             
│   ├── group1-shard2of15.bin                           
│   ├── ....                       
│   ├── group1-shard15of15.bin  
│   └── model.json
├── tflite/
│   ├── model.tflite                                    
│   └── label.txt
├── notebook.ipynb                                   
├── README.md
└── requirements.txt                          
```

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/FR21/RiceLeaf_DiseaseDetection.git
    cd RiceLeaf_DiseaseDetection
    ```

2. Install the required dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Train the Model**: To train the rice leaf disease detection model, run the `RiceLeaf_DiseaseDetection.ipynb` notebook. This notebook provides a complete guide for training, evaluating, and running inference with the model.
   
2. **Use the TFLite Model**: The trained model can be deployed to mobile devices using the **TFLite** format found in the `tflite/` folder.

3. **Use the TensorFlow.js Model**: For web applications, you can use the model available in the `tfjs_model/` folder.

## Contribution

If you are interested in contributing to this project, please follow these steps:
1. Fork this repository.
2. Create a new branch for the feature or fix you want to implement.
3. Submit a pull request once you've completed the changes.

---

Thank you for using this project! We hope it helps improve the efficiency of agriculture!

