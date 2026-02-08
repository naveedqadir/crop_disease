"""
Model Loader Module

This module handles loading and managing the pretrained deep learning model
for crop disease detection using HuggingFace transformers.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Import HuggingFace model config from settings
try:
    from config.settings import HUGGINGFACE_MODEL
except ImportError:
    # Fallback if settings not available
    HUGGINGFACE_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


class CropDiseaseModel:
    """
    Wrapper class for the crop disease detection model.
    Uses HuggingFace transformers for production inference.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        class_labels: Optional[List[str]] = None,
        framework: str = "tensorflow"
    ):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to custom model file (optional)
            class_labels: List of class label names
            framework: Deep learning framework (default uses HuggingFace)
        """
        self.model_path = model_path
        self.class_labels = class_labels
        self.framework = framework.lower()
        self.model = None
        self.processor = None
        self.input_size = (224, 224)
        self.use_huggingface = True
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the pretrained model from HuggingFace or local file.
        
        Args:
            model_path: Path to custom model file (optional)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        path = model_path or self.model_path
        
        # Try loading custom model if path provided
        if path and os.path.exists(path):
            return self._load_custom_model(path)
        
        # Default: Load HuggingFace production model
        return self._load_huggingface_model()
    
    def _load_huggingface_model(self) -> bool:
        """Load the pre-trained HuggingFace model for plant disease detection."""
        try:
            from transformers import AutoModelForImageClassification
            import torch
            
            logger.info(f"Loading HuggingFace model: {HUGGINGFACE_MODEL}")
            
            # Load model only (handle preprocessing manually for MobileNet)
            self.model = AutoModelForImageClassification.from_pretrained(HUGGINGFACE_MODEL)
            self.model.eval()
            
            # Get model's class labels
            self.hf_labels = self.model.config.id2label
            
            logger.info(f"HuggingFace model loaded successfully ({len(self.hf_labels)} classes)")
            logger.info(f"Model accuracy: 95.4% on PlantVillage dataset")
            self.use_huggingface = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            logger.info("Falling back to untrained TensorFlow model...")
            return self._create_fallback_model()
    
    def _load_custom_model(self, path: str) -> bool:
        """Load a custom TensorFlow/PyTorch model from file."""
        try:
            if self.framework == "tensorflow":
                import tensorflow as tf
                self.model = tf.keras.models.load_model(path)
                logger.info(f"Loaded custom TensorFlow model from {path}")
            elif self.framework == "pytorch":
                import torch
                self.model = torch.load(path)
                self.model.eval()
                logger.info(f"Loaded custom PyTorch model from {path}")
            self.use_huggingface = False
            return True
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            return self._load_huggingface_model()
    
    def _create_fallback_model(self) -> bool:
        """Create an untrained fallback model if HuggingFace model fails to load."""
        try:
            import tensorflow as tf
            
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            num_classes = len(self.class_labels) if self.class_labels else 38
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            self.use_huggingface = False
            logger.warning("Using untrained fallback model - predictions will be inaccurate. Please check internet connection.")
            return True
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for model inference.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image ready for model input
        """
        import cv2
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (MobileNetV2 input size)
        resized = cv2.resize(rgb_image, self.input_size)
        
        if self.use_huggingface:
            import torch
            # Normalize to [0, 1] then apply ImageNet normalization
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            # Convert to PyTorch format: (C, H, W) and add batch dimension
            tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
            return tensor
        else:
            # TensorFlow preprocessing
            normalized = resized.astype(np.float32) / 255.0
            return np.expand_dims(normalized, axis=0)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Make a prediction on an input image using real ML inference.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Tuple of (predicted_class, confidence, top_5_predictions)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        preprocessed = self.preprocess_image(image)
        
        if self.use_huggingface:
            return self._predict_huggingface(preprocessed)
        else:
            return self._predict_tensorflow(preprocessed)
    
    def _predict_huggingface(self, tensor) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Run inference using HuggingFace model."""
        import torch
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(tensor)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get top predictions
        top_k = min(5, len(probabilities))
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            # Get label from model config
            hf_label = self.hf_labels[idx.item()]
            
            # Map to our class labels if available
            label = self._map_label(hf_label)
            confidence = float(prob.item())
            top_predictions.append((label, confidence))
        
        best_label, best_confidence = top_predictions[0]
        return best_label, best_confidence, top_predictions
    
    def _predict_tensorflow(self, preprocessed: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Run inference using TensorFlow model."""
        predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        top_indices = np.argsort(predictions)[::-1][:5]
        
        top_predictions = []
        for idx in top_indices:
            if self.class_labels and idx < len(self.class_labels):
                label = self.class_labels[idx]
            else:
                label = f"Class_{idx}"
            confidence = float(predictions[idx])
            top_predictions.append((label, confidence))
        
        best_label, best_confidence = top_predictions[0]
        return best_label, best_confidence, top_predictions
    
    def _map_label(self, hf_label: str) -> str:
        """
        Map HuggingFace model label to our class label format.
        HuggingFace uses spaces, we use underscores (PlantVillage format).
        """
        # Convert spaces to underscores and standardize format
        mapped = hf_label.replace(" ", "_").replace(",", "")
        
        # Try to find matching label in our class labels
        if self.class_labels:
            for class_label in self.class_labels:
                # Normalize both for comparison
                if self._normalize_label(class_label) == self._normalize_label(mapped):
                    return class_label
        
        return mapped
    
    def _normalize_label(self, label: str) -> str:
        """Normalize a label for comparison."""
        return label.lower().replace("_", "").replace(" ", "").replace("(", "").replace(")", "").replace(",", "")
    
    def load_class_labels(self, labels_path: str) -> bool:
        """
        Load class labels from a JSON file.
        
        Args:
            labels_path: Path to the JSON file containing class labels
            
        Returns:
            True if labels loaded successfully, False otherwise
        """
        try:
            with open(labels_path, 'r') as f:
                self.class_labels = json.load(f)
            logger.info(f"Loaded {len(self.class_labels)} class labels from {labels_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load class labels: {e}")
            return False


def create_model(
    model_path: Optional[str] = None,
    class_labels: Optional[List[str]] = None,
    framework: str = "tensorflow"
) -> CropDiseaseModel:
    """
    Factory function to create and load a crop disease detection model.
    
    Args:
        model_path: Path to custom model file (optional)
        class_labels: List of class label names
        framework: Deep learning framework ('tensorflow' or 'pytorch')
        
    Returns:
        Loaded CropDiseaseModel instance
    """
    model = CropDiseaseModel(
        model_path=model_path,
        class_labels=class_labels,
        framework=framework
    )
    model.load_model()
    return model
