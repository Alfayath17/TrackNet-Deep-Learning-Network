# TrackNet – Deep Learning Network

A deep learning-based computer vision system for **high-speed object tracking, localization, trajectory analysis, and bounce detection from video**.

This project implements a TrackNet-based neural network to detect small and fast-moving objects across consecutive video frames. The predicted object coordinates can be used to generate trajectories and analyze movement patterns from video data.

---

## 📌 Project Overview

Tracking small and fast-moving objects in video is challenging because the object can occupy only a small number of pixels, move rapidly between frames, and become difficult to distinguish from the background.

This project uses a **TrackNet deep learning architecture** to process video frames and predict the spatial location of the target object.

### Key capabilities

- Video frame processing
- Object localization
- Coordinate prediction
- Trajectory generation
- Ground-truth generation
- Object tracking
- Bounce detection
- Model training
- Model testing
- Video inference
- Tracking result visualization

---

## 🎯 Objectives

The main objectives of this project are:

- Detect small and fast-moving objects in video.
- Predict the object's position in individual frames.
- Track the object across consecutive frames.
- Generate ground-truth data for training and evaluation.
- Generate object trajectories from predicted coordinates.
- Identify movement and bounce patterns.
- Evaluate the trained deep learning model.
- Generate an output video with tracking results.

---

## 🧠 TrackNet Architecture

TrackNet is designed for detecting small objects in video by learning spatial and temporal information from consecutive frames.

### Workflow

```text
                 Input Video
                     │
                     ▼
              Frame Extraction
                     │
                     ▼
             Data Preprocessing
                     │
                     ▼
          Consecutive Video Frames
                     │
                     ▼
              TrackNet Model
                     │
                     ▼
           Object Localization
                     │
                     ▼
          X / Y Coordinate Output
                     │
             ┌───────┴────────┐
             ▼                ▼
       Trajectory          Bounce
        Analysis          Detection
             │                │
             └───────┬────────┘
                     ▼
              Tracking Output
                     │
                     ▼
             Processed Video
```

---

## 🔬 Project Workflow

### 1. Dataset Preparation

Video and image data are prepared for training and testing.

The project contains a dedicated:

```text
Dataset/
```

directory for input data.

### 2. Ground Truth Generation

The `gt_gen.py` module is used to generate ground-truth information required for model training and evaluation.

```text
Video / Frames
       ↓
Ground Truth Generation
       ↓
GT_Output/
       ↓
Training / Evaluation
```

### 3. Model Training

The TrackNet model is implemented in:

```text
model.py
```

The main training workflow is handled through:

```text
main.py
```

Additional bounce-related training functionality is implemented in:

```text
bounce_train.py
```

### 4. Object Detection and Inference

The inference functionality is implemented in:

```text
infer_on_video.py
```

The model processes video frames and generates predicted object coordinates.

### 5. Trajectory Generation

Predicted coordinates can be used to reconstruct the object's movement through the video.

Example:

```text
Frame      X       Y
----------------------
001       320     185
002       328     190
003       337     196
004       345     203
005       352     211
```

These coordinates form the object's movement trajectory.

### 6. Bounce Detection

The project also includes bounce-related analysis.

The bounce training / analysis functionality is handled by:

```text
bounce_train.py
ctb_regr_bounce.py
```

This can be used to analyze changes in the object's trajectory and identify potential bounce events.

---

## 📊 Input and Output

### Input

The system accepts video data containing the target moving object.

Example:

```text
test_video.mp4
```

### Output

The system generates tracking information and processed video output.

Example:

```text
tracked_output.mp4
```

The output can contain the detected object location and tracking trajectory over the video frames.

---

## 📁 Project Structure

```text
TrackNet-Deep-Learning-Network/
│
├── Dataset/
│   └── Training / Testing Data
│
├── GT_Output/
│   └── Generated Ground Truth
│
├── exps/
│   └── default/
│       ├── plots/
│       └── model_last.pth
│
├── bounce_train.py
├── ctb_regr_bounce.py
├── datasets.py
├── gt_gen.py
├── infer_on_video.py
├── main.py
├── model.py
├── test.py
│
├── test_video.mp4
├── tracked_output.mp4
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **Deep Learning**
- **Computer Vision**
- **TrackNet**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Video Processing**
- **Neural Networks**

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Alfayath17/TrackNet-Deep-Learning-Network.git
```

Navigate to the project directory:

```bash
cd TrackNet-Deep-Learning-Network
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the Model

```bash
python main.py
```

### Generate Ground Truth

```bash
python gt_gen.py
```

### Test the Model

```bash
python test.py
```

### Run Inference on Video

```bash
python infer_on_video.py
```

### Bounce Analysis

```bash
python bounce_train.py
```

> The exact arguments or configuration may depend on the current implementation and dataset setup.

---

## 📈 Results and Analysis

The model produces object coordinates from video frames, which can be used to visualize movement trajectories.

Example:

```text
              ●
             /
            ●
           /
          ●
         /
        ●
       /
      ●
```

The predicted coordinates can be further analyzed to determine:

- Object position
- Movement direction
- Trajectory
- Distance traveled
- Movement patterns
- Bounce events

---

## 🎥 Video Tracking

The project includes a sample test video:

```text
test_video.mp4
```

After inference, a processed tracking video can be generated as:

```text
tracked_output.mp4
```

This provides a visual representation of the model's tracking performance.

---

## 📊 Experiment Tracking

Model experiments and generated training information are stored under:

```text
exps/default/
```

The experiment directory can contain:

- Model checkpoints
- Training plots
- Evaluation results
- Training logs

Example:

```text
exps/
└── default/
    └── plots/
```

---

## 🔍 Core Python Modules

| File | Purpose |
|---|---|
| `model.py` | TrackNet deep learning model architecture |
| `datasets.py` | Dataset loading and preprocessing |
| `gt_gen.py` | Ground-truth generation |
| `main.py` | Main training workflow |
| `test.py` | Model testing and evaluation |
| `infer_on_video.py` | Video inference and object tracking |
| `bounce_train.py` | Bounce-related training/analysis |
| `ctb_regr_bounce.py` | Bounce regression / analysis functionality |

---

## 🚀 Applications

### Sports Analytics

Potential applications include tracking:

- Tennis balls
- Cricket balls
- Shuttlecocks
- Other small fast-moving objects

### Computer Vision

- Object localization
- Object tracking
- Movement analysis
- Video analytics

### Performance Analysis

Tracking data can be used to calculate:

- Position
- Distance
- Speed
- Direction
- Trajectory
- Bounce locations

---

## 🔮 Future Improvements

Possible future improvements include:

- Real-time object tracking
- GPU-optimized inference
- Improved detection accuracy
- Multi-object tracking
- Automatic speed calculation
- Advanced trajectory analysis
- Improved bounce detection
- Web-based visualization dashboard
- REST API integration
- Deployment as a computer vision service
- Integration with sports analytics systems

---

## 📚 Key Learning Areas

This project provides practical experience in:

- Deep Learning
- Computer Vision
- CNN-based architectures
- Video processing
- Object localization
- Object tracking
- Dataset preparation
- Ground-truth generation
- Model training
- Model evaluation
- Trajectory analysis
- Python development

---

## 👨‍💻 Author

### Alfayath

**Data Analyst | Python | SQL | Power BI | AI & Machine Learning**

GitHub:  
https://github.com/Alfayath17

---

## ⭐ Repository

If you find this project useful for learning, research, or experimentation, consider giving the repository a ⭐.

---

## 📄 License

This project is intended for educational, research, and experimental purposes.
