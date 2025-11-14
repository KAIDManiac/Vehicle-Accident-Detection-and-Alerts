# Crash Detection System Using CNN and YOLOv8

An AI-based crash detection system that combines **Convolutional Neural Networks (CNNs)** and **YOLOv8 object tracking** to detect vehicle crashes from video feeds in real time.

---

## Overview

This project automatically detects car accidents from videos.  
It uses:
- **YOLOv8** for vehicle detection and tracking  
- A custom-trained **CNN classifier** to distinguish accident vs. non-accident frames  
- Frame-by-frame logic to confirm crashes with persistence and motion analysis  

---

## Project Structure

```

Crash Detection/
├── annotations/          # Optional labels or metadata
├── test_src/             # Main source code & notebooks
│   ├── data_prep2.ipynb      # Dataset creation and preprocessing
│   ├── data_train.ipynb      # CNN model training and evaluation
│   └── detect_videoo.ipynb   # Real-time crash detection on videos
├── Test Dataset/         # Test videos, dataset folders, trained model
│   ├── 0/                    # Non-accident images
│   ├── 1/                    # Accident images
│   ├── train_split.csv       # Training data metadata
│   ├── test_split.csv        # Testing data metadata
│   └── crash_cnn_model.h5    # Trained CNN model
└── .vscode/              # Editor configuration

````

---

## Requirements

Install dependencies with:

```bash
pip install tensorflow keras opencv-python ultralytics scikit-learn matplotlib pandas numpy pillow
````

Optional (to run notebooks):

```bash
pip install jupyter
```

---

## Workflow

### 1️ Dataset Preparation — `data_prep2.ipynb`

* Extracts frames from crash and non-crash videos.
* Labels them as:

  * `0` → non-accident
  * `1` → accident
* Splits data (80% train / 20% test).
* Saves `train_split.csv` and `test_split.csv`.

### 2 Model Training — `data_train.ipynb`

* Loads training and test images from the dataset.
* Builds a CNN with 3 convolutional blocks:

  * Conv → MaxPool → Flatten → Dense → Dropout → Softmax
* Trains for 10 epochs using Adam optimizer and categorical cross-entropy.
* Evaluates accuracy and loss.
* Saves model as `crash_cnn_model.h5`.

### 3 Video Crash Detection — `detect_videoo.ipynb`

* Loads YOLOv8 for vehicle detection.
* Loads trained CNN model for crash classification.
* Tracks vehicles with unique IDs.
* Computes crash confidence across multiple frames for reliability.
* Highlights:

  * 🔵 SAFE
  * 🟡 INITIALIZING / MOVING AWAY
  * 🔴 CRASH CONFIRMED
* Displays video frames and extracts cropped crash images.

---

## Results

| Metric              | Value     |
| ------------------- | --------- |
| Accuracy            | **97.5%** |
| Precision           | **98.5%** |
| Recall              | **96.6%** |
| F1-Score            | **97.5%** |
| False Positive Rate | **1.5%**  |

Model performs efficiently with low latency (~0.87 ms/frame).

---

##  How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/Crash-Detection.git
   cd Crash-Detection
   ```

2. **Run the notebooks or scripts**

   * Prepare dataset:

     ```bash
     jupyter notebook test_src/data_prep2.ipynb
     ```
   * Train model:

     ```bash
     jupyter notebook test_src/data_train.ipynb
     ```
   * Detect crashes:

     ```bash
     jupyter notebook test_src/detect_videoo.ipynb
     ```

3. **Run on your own video**

   * Replace `VIDEO_PATH` in `detect_videoo.ipynb` with your video file path.
   * Run all cells.

# Key Features

* Real-time crash detection from video feeds
* Combined CNN + YOLOv8 for enhanced accuracy
* Vehicle tracking and motion-based crash confirmation
* Automatic extraction of crash frames
* Visual overlay of crash status on each vehicle


## Future Improvements

* Integrate real-time alerting (SMS/Email/API).
* Deploy on edge devices or Raspberry Pi.
* Expand dataset with varied weather & lighting conditions.



# License

This project is open-source under the **MIT License**.

---

# Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for vehicle detection.
* TensorFlow / Keras for CNN training.
* OpenCV for video processing.


# Author

**Syed Abubaker Ahmed**

AI/ML Engineer | Computer Vision Developer | Autonomous Systems & Smart Mobility | Smart City Technology Researcher | CS Undergraduate
