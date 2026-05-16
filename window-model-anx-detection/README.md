# Window-Based Anxiety Detection
  
This module implements a window-based data loading pipeline for anxiety detection using multimodal DVlog features.   
The previous implementation used complete sequences, whereas this updated version divides the temporal features into overlapping fixed-size windows for better temporal learning and increased training samples.  
  
## Features  
  
- Loads visual and acoustic feature embeddings  
- Concatenates multimodal features  
- Creates sliding temporal windows  
- Supports configurable:  
  - window size  
  - overlap ratio  
  - batch size  
  - gender filtering  
- Compatible with PyTorch `DataLoader`  
  
## Windowing Strategy  
  
Given a sequence of length `T`:  
  
- Fixed-size windows are extracted using a sliding window approach  
- Window step size is computed as:  
  
```python  
step = window_size * (1 - overlap)  
