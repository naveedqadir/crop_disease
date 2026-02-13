# Crop Disease Detection System

A real-time video analytics system for crop disease detection using deep learning. This system processes video frames from webcam, Raspberry Pi camera, or pre-recorded video files, predicts crop diseases using a production-grade HuggingFace model, and displays results in real-time.

## Model

This system uses **linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification** from HuggingFace:
- **Accuracy**: 95.4% on PlantVillage dataset
- **Architecture**: MobileNetV2 (optimized for mobile/edge devices)
- **Classes**: 38 plant disease categories
- **Training Data**: PlantVillage dataset with 54,000+ images

## Features

- **Production ML Inference**: Real predictions from trained HuggingFace model (95.4% accuracy)
- **Multi-source video input**: Webcam, Raspberry Pi camera, or video files
- **Image analysis**: Process single images or entire folders of images
- **Real-time disease detection**: Uses pretrained deep learning models
- **Cross-platform support**: Works on PC/laptop and Raspberry Pi
- **Live statistics dashboard**: Real-time tracking of predictions, disease rates, confidence scores
- **Excel report export**: Automatic generation of detailed Excel reports for each session
- **Configurable settings**: Frame rate, resolution, skip frames for performance
- **Visual feedback**: Annotated video display with prediction confidence
- **Screenshot capture**: Save frames with predictions
- **Headless mode**: Run without display for automated processing

## Supported Crops & Diseases

The system is trained on the PlantVillage dataset and can detect diseases in:

- **Apple**: Scab, Black Rot, Cedar Apple Rust
- **Cherry**: Powdery Mildew
- **Corn**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight
- **Grape**: Black Rot, Esca, Leaf Blight
- **Orange**: Citrus Greening
- **Peach**: Bacterial Spot
- **Pepper**: Bacterial Spot
- **Potato**: Early Blight, Late Blight
- **Squash**: Powdery Mildew
- **Strawberry**: Leaf Scorch
- **Tomato**: Multiple diseases including Bacterial Spot, Early/Late Blight, Leaf Mold, etc.

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or video file for testing
- (Optional) Raspberry Pi with camera module

### Setup

1. **Clone or create the project directory**:
   ```bash
   cd crop_disease
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
 
   # Windows
   venv\Scripts\activate
 
   # Linux/macOS
   source venv/bin/activate
   ```
 
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
 
4. **For Raspberry Pi** (additional steps):
   ```bash
   # Install picamera2
   sudo apt install -y python3-picamera2
 
   # Or via pip
   pip install picamera2
 
   pip install --upgrade pip setuptools wheel
   pip install -r requirements_rasp.txt
 
  sudo apt --fix-broken install
  #if needed
  sudo apt-get update
  sudo apt-get upgrade
 
  sudo apt-get install libcap-dev
  pip install picamera2
 
 
   ```

## Usage

### Basic Usage

```bash
# Development mode: Uses test video
python main.py

# Production mode: Use webcam/camera
python main.py --camera

# Use a specific camera (by index)
python main.py --source 1

# Process a video file
python main.py --source path/to/video.mp4

# Use Raspberry Pi camera
python main.py --picamera

# Test with tomato video
python main.py --source data/videos/tomato_fruit_real.mp4

# Analyze a single image
python main.py --image path/to/plant_image.jpg

# Analyze all images in a folder
python main.py --image data/images/
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--image` | `-i` | Analyze image file or folder of images | `None` |
| `--source` | `-s` | Video source (camera index or file path) | Test video |
| `--camera` | | Use webcam for live video (production mode) | `False` |
| `--picamera` | `-p` | Use Raspberry Pi camera | `False` |
| `--model` | `-m` | Path to custom model file (overrides HuggingFace) | `None` |
| `--framework` | `-f` | Framework for custom models (tensorflow/pytorch) | `tensorflow` |
| `--width` | `-W` | Frame width | `640` |
| `--height` | `-H` | Frame height | `480` |
| `--skip-frames` | `-k` | Process every nth frame | `5` |
| `--confidence-threshold` | `-c` | Min confidence to display | `0.5` |
| `--log-level` | `-l` | Logging verbosity | `INFO` |
| `--no-display` | | Run in headless mode | `False` |
| `--save-output` | `-o` | Save output video to file | `None` |
| `--no-excel` | | Disable Excel report generation | `False` |
| `--excel-dir` | | Custom directory for Excel reports | `reports/` |

### Examples

```bash
# Development: Test with leaf disease video
python main.py

# Test with tomato fruit video
python main.py --source data/videos/tomato_fruit_real.mp4

# Production: Use live camera
python main.py --camera

# High-quality processing (all frames)
python main.py --camera --skip-frames 1 --width 1280 --height 720

# Save output video
python main.py --source input.mp4 --save-output output.mp4

# Disable Excel export
python main.py --no-excel

# Custom Excel output directory
python main.py --excel-dir ./my_reports

# Analyze single image
python main.py --image plant_photo.jpg

# Analyze all images in folder
python main.py --image data/images --confidence-threshold 0.3

# Raspberry Pi with optimized settings (auto-applied)
python main.py --picamera

# Raspberry Pi headless mode for automation
python main.py --picamera --no-display

# Use custom trained model
python main.py --model models/custom_model.pth --framework pytorch

# Debug mode
python main.py --log-level DEBUG
```

### Keyboard Controls

- **Q**: Quit the application
- **S**: Save screenshot to `screenshots/` directory

## Project Structure

```
crop_disease/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore patterns
├── config/
│   ├── __init__.py
│   └── settings.py        # Configuration settings
├── src/
│   ├── __init__.py
│   ├── model_loader.py    # Production ML inference with HuggingFace model
│   ├── video_processor.py # Video capture and display
│   ├── statistics_tracker.py # Live statistics tracking
│   └── data_exporter.py   # Excel report generation
├── models/                 # Custom models directory (optional)
│   └── __init__.py
├── data/
│   ├── videos/            # Test videos
│   │   ├── plant_disease_real.mp4  # Leaf disease video
│   │   ├── tomato_fruit_real.mp4   # Tomato fruit video
│   │   └── apple_tree_fruit.mp4    # Apple tree video
│   └── images/            # Test images
│       ├── disease_leaf_spots.jpg
│       ├── yellow_leaf_infection.jpg
│       ├── plant_rust_spots.jpg
│       ├── green_leaf_dark_spots.jpg
│       └── leaf_brown_decay.jpg
├── logs/                   # Application logs
├── reports/                # Excel detection reports (auto-generated)
└── screenshots/            # Saved screenshots
```

## Test Videos

The system includes three test videos downloaded from Pexels:

1. **plant_disease_real.mp4** - Leaves with powdery mildew (white fungal spots)
2. **tomato_fruit_real.mp4** - Ripe tomato fruits on the vine
3. **apple_tree_fruit.mp4** - Apple tree with fruit (tests disease detection on apples)

## Test Images

The `data/images/` folder contains sample plant disease images for testing:

1. **disease_leaf_spots.jpg** - Green leaves with dry spots and discoloration
2. **yellow_leaf_infection.jpg** - Yellow leaf with dark spots showing infection
3. **plant_rust_spots.jpg** - Leaf with rust spots (fungal disease)
4. **green_leaf_dark_spots.jpg** - Green leaves with dark spot patterns
5. **leaf_brown_decay.jpg** - Leaf with brown spots showing decay

### Image Mode Usage

```bash
# Analyze a single image
python main.py --image data/images/disease_leaf_spots.jpg

# Analyze all images in the folder
python main.py --image data/images/

# With lower confidence threshold to see more predictions
python main.py --image data/images/ --confidence-threshold 0.2 --no-display
```

The production model uses deep learning to classify plant diseases based on visual features learned from the PlantVillage dataset.

## Excel Reports

Each session automatically generates an Excel report containing:

### Summary Sheet
- Session information (start time, video source, model, settings)
- Detection statistics (total predictions, disease rate, confidence)
- Prediction distribution table

### Detections Sheet
- Frame-by-frame detection results
- Timestamp, prediction, confidence, status
- Color-coded rows (green = healthy, red = disease)
- Top alternative predictions

### Raw Data Sheet
- Complete raw data for custom analysis
- All recorded fields in tabular format

**Example report filename**: `detection_results_sample_crop_test_20260208_181530.xlsx`

**Location**: `reports/` directory (or custom via `--excel-dir`)

## Using Custom Models

By default, the system uses the pre-trained HuggingFace model. You can optionally use your own trained models:

### PyTorch Model

```python
# Save your PyTorch model
torch.save(model.state_dict(), 'models/crop_disease_model.pth')

# Use it with the application
python main.py --framework pytorch --model models/crop_disease_model.pth
```

### TensorFlow/Keras Model (Fallback)

```python
# Train and save your model
model.save('models/crop_disease_model.h5')

# Use it with the application
python main.py --model models/crop_disease_model.h5
```

### Model Requirements

- **Input size**: 224x224 RGB images
- **Output**: Softmax probabilities for 38 classes (PlantVillage dataset)

## Training Your Own Model

For training a custom model on the PlantVillage dataset:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

2. Use transfer learning with MobileNetV2 (PyTorch):

```python
import torch
import torchvision.models as models
import torch.nn as nn

# Load pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Replace classifier for 38 classes
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 38)
)

# Train on your data
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop...
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), 'models/crop_disease_model.pth')
```

## Performance Optimization

### For PC/Laptop

- Use GPU acceleration with PyTorch CUDA
- Increase `--skip-frames` for slower hardware
- Reduce resolution for faster processing

### For Raspberry Pi

- Use `--picamera` flag - automatically applies optimized settings (skip_frames=10)
- For even lower resource usage: `--width 320 --height 240`
- Use headless mode for automated processing: `--no-display`
- The system auto-detects Raspberry Pi and uses picamera2 library

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   ```bash
   # Check available cameras
   python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
   ```

2. **Model download issues**:
   ```bash
   # Check internet connection - model downloads from HuggingFace on first run
   pip install --upgrade transformers torch
   ```

3. **Pi Camera not working**:
   ```bash
   # Enable camera in raspi-config
   sudo raspi-config
   # Navigate to Interface Options > Camera > Enable
   ```

4. **Out of memory on Raspberry Pi**:
   - Increase swap space
   - Reduce frame resolution
   - Use `--skip-frames 10` or higher

## License

This project is for educational purposes. The PlantVillage dataset is available under CC0 license.

## Acknowledgments

- [PlantVillage Dataset](https://plantvillage.psu.edu/)
- [HuggingFace Transformers](https://huggingface.co/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- Model by [linkanjarad](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification)
