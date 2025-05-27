# Emotion Detection from Images using CNN

## 📌 Project Overview
This project focuses on detecting human emotions from facial images using Convolutional Neural Networks (CNN). Trained on the FER-2013 dataset, the system can classify seven basic emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

## 🧑‍💻 Group Information
- **Group No**: 8  
- **Group Name**: leisurely_loco

## 👥 Team Members
- Arbin Zaman (ID: 2125051006)
- Sohana Afrin (ID: 2125051013)
- Safin Ahamed Sajid (ID: 2125051022)
- Md. Fahim Abrar Asif (ID: 2125051116)

## 🎯 Objectives
- Build a CNN model to recognize facial emotions
- Use the FER-2013 dataset for training and validation
- Enable emotion detection in real-time from static images or webcam
- Evaluate model performance and optimize for accuracy and generalization

## ⚙️ Tools and Environment
- **Language**: Python  
- **IDE**: Google Colab  
- **Libraries**: TensorFlow/Keras, OpenCV, Matplotlib, NumPy  
- **Dataset**: FER-2013 (from Kaggle)

## 🔧 Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/arbinzaman/Emotion-Detection-System.git
    cd Emotion-Detection-System
    ```

2. Install required libraries:
    ```bash
    pip install tensorflow opencv-python matplotlib numpy
    ```

3. Download FER-2013 dataset:
    - Register and accept terms on Kaggle.
    - Use `kaggle datasets download -d msambare/fer2013` or manually upload the dataset to Colab.

## 🧠 Model Architecture
The CNN consists of:
- 2 convolutional layers with ReLU activation and max pooling
- Dropout layers to reduce overfitting
- Dense layers with softmax for emotion classification (7 classes)

## 🚀 How to Run
Run the Jupyter Notebook (`emotion_detection_colab.ipynb`) step-by-step:
- Preprocess dataset
- Train CNN model
- Use `predict_emotion(img_path)` function to test images

## 📸 Sample Output

Input Image | Predicted Emotion
------------|------------------
![input](examples/input.png) | 😄 Happy

## 📈 Results
- Validation Accuracy: ~60%  
- Real-time predictions on unseen images work reliably under various lighting conditions  
- Robust against moderate noise and variations

## 🧪 Limitations
- Struggles with underrepresented emotions like "Disgust" and "Fear"
- Grayscale images only – lacks color cues
- Sensitive to extreme lighting or occluded faces

## 🔮 Future Work
- Use colored, high-resolution images
- Add real-time webcam support
- Deploy via TensorFlow Lite for mobile apps
- Integrate multimodal data (voice/text)

## 📚 References
1. [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  
2. François Chollet, *Deep Learning with Python*, Manning, 2017  
3. [TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)  
4. [OpenCV Documentation](https://docs.opencv.org/)

## 🔗 GitHub Repository
[https://github.com/arbinzaman/Emotion-Detection-System](https://github.com/arbinzaman/Emotion-Detection-System)
