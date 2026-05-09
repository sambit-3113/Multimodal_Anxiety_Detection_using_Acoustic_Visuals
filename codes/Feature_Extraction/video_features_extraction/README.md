# Video Feature Extraction

This repository contains scripts for extracting facial landmark-based visual features from video data using dlib's 68-point facial landmark detector.

## Feature Types

Depending on the experiment or model requirement, different temporal feature representations can be used:

- **1-second features (1s):**  
  Extract features from one frame per second.

- **0.5-second averaged features (0.5s averaged):**  
  Average landmark features over 0.5-second windows for smoother temporal representation.

- **Framewise raw features:**  
  Extract landmark coordinates from every frame without temporal averaging.

## Output

The extracted features are saved as:
- `.csv` files
- `.npy` NumPy arrays

## Download the Model

The trained model file is too large to store in this repository.

Please download it from:

(https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat)

After downloading, place the model file inside:

```text
models/
