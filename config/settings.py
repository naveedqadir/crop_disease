"""
Configuration settings for the Crop Disease Detection System.
Production model: HuggingFace linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# HuggingFace Model Configuration (Production)
HUGGINGFACE_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# Default class labels for PlantVillage dataset (38 classes)
DEFAULT_CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Test video path
TEST_VIDEO_PATH = DATA_DIR / "videos" / "plant_disease_real.mp4"

# Video settings
VIDEO_SETTINGS = {
    "default_source": str(TEST_VIDEO_PATH),
    "camera_source": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "skip_frames": 5,
}

# Display settings
DISPLAY_SETTINGS = {
    "window_name": "Crop Disease Detection",
    "font_scale": 0.7,
    "font_thickness": 2,
    "box_color": (0, 255, 0),
    "text_color": (255, 255, 255),
    "background_color": (0, 0, 0),
    "confidence_threshold": 0.5,
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": BASE_DIR / "logs" / "app.log",
}

# Raspberry Pi specific settings (used with --picamera flag)
RASPBERRY_PI_SETTINGS = {
    "resolution": (640, 480),
    "framerate": 30,
    "skip_frames": 10,  # Higher skip for Pi's limited CPU
}
