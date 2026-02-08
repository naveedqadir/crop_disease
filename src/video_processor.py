"""
Video Processor Module

This module handles video capture from various sources including webcam,
Raspberry Pi camera, and video files.
"""

import cv2
import logging
import time
import platform
from typing import Optional, Tuple, Generator
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video capture."""
    source: any = 0  # Can be int (camera index) or str (file path)
    width: int = 640
    height: int = 480
    fps: int = 30
    use_picamera: bool = False
    skip_frames: int = 1


class VideoProcessor:
    """
    Video processor class that handles video capture from multiple sources.
    Supports webcam, Raspberry Pi camera, and video files.
    """
    
    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize the video processor.
        
        Args:
            config: VideoConfig object with capture settings
        """
        self.config = config or VideoConfig()
        self.cap = None
        self.picamera = None
        self.is_running = False
        self.frame_count = 0
        self.fps_start_time = None
        self.current_fps = 0.0
        
        # Detect if running on Raspberry Pi
        self.is_raspberry_pi = self._detect_raspberry_pi()
        
    def _detect_raspberry_pi(self) -> bool:
        """Detect if the system is a Raspberry Pi."""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'Raspberry Pi' in model
        except:
            return False
    
    def initialize(self) -> bool:
        """
        Initialize the video capture device.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self.config.use_picamera and self.is_raspberry_pi:
                return self._init_picamera()
            else:
                return self._init_opencv()
        except Exception as e:
            logger.error(f"Failed to initialize video capture: {e}")
            return False
    
    def _init_opencv(self) -> bool:
        """Initialize OpenCV video capture."""
        self.cap = cv2.VideoCapture(self.config.source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.config.source}")
            return False
        
        # Set video properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"OpenCV video capture initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        self.is_running = True
        return True
    
    def _init_picamera(self) -> bool:
        """Initialize Raspberry Pi camera."""
        try:
            from picamera2 import Picamera2
            
            self.picamera = Picamera2()
            config = self.picamera.create_preview_configuration(
                main={"size": (self.config.width, self.config.height), "format": "RGB888"}
            )
            self.picamera.configure(config)
            self.picamera.start()
            
            logger.info(f"Pi Camera initialized: {self.config.width}x{self.config.height}")
            self.is_running = True
            return True
            
        except ImportError:
            logger.warning("picamera2 not available. Falling back to OpenCV.")
            return self._init_opencv()
        except Exception as e:
            logger.error(f"Failed to initialize Pi Camera: {e}")
            return self._init_opencv()
    
    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """
        Read a single frame from the video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_running:
            return False, None
        
        try:
            if self.picamera is not None:
                frame = self.picamera.capture_array()
                # Convert RGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            elif self.cap is not None:
                ret, frame = self.cap.read()
                return ret, frame
            else:
                return False, None
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def get_frames(self) -> Generator:
        """
        Generator that yields frames from the video source.
        
        Yields:
            Tuple of (frame, frame_number, should_process)
        """
        self.fps_start_time = time.time()
        frame_times = []
        
        while self.is_running:
            ret, frame = self.read_frame()
            
            if not ret:
                if isinstance(self.config.source, str):
                    # Video file ended
                    logger.info("Video file playback completed")
                    break
                else:
                    # Camera error
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
            
            self.frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            frame_times.append(current_time)
            
            # Keep only last 30 frame times for FPS calculation
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            if len(frame_times) > 1:
                self.current_fps = len(frame_times) / (frame_times[-1] - frame_times[0])
            
            # Determine if this frame should be processed for inference
            should_process = (self.frame_count % self.config.skip_frames) == 0
            
            yield frame, self.frame_count, should_process
    
    def get_fps(self) -> float:
        """Get the current FPS."""
        return self.current_fps
    
    def get_frame_count(self) -> int:
        """Get the total number of frames processed."""
        return self.frame_count
    
    def get_video_info(self) -> dict:
        """Get information about the video source."""
        if self.cap is not None:
            return {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "codec": int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            }
        elif self.picamera is not None:
            return {
                "width": self.config.width,
                "height": self.config.height,
                "fps": self.config.fps,
                "total_frames": -1,
                "codec": "picamera",
            }
        return {}
    
    def release(self):
        """Release video capture resources."""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("OpenCV video capture released")
            
        if self.picamera is not None:
            self.picamera.stop()
            self.picamera = None
            logger.info("Pi Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class DisplayManager:
    """Manages the display window and rendering of results with live statistics."""
    
    # Color scheme
    COLORS = {
        "healthy": (0, 200, 0),       # Green
        "disease": (0, 0, 220),       # Red
        "warning": (0, 180, 255),     # Orange
        "info": (255, 200, 0),        # Cyan
        "text": (255, 255, 255),      # White
        "text_dim": (180, 180, 180),  # Gray
        "bg_dark": (30, 30, 30),      # Dark gray
        "bg_panel": (45, 45, 45),     # Panel background
        "accent": (255, 100, 50),     # Blue accent
    }
    
    def __init__(
        self,
        window_name: str = "Crop Disease Detection",
        font_scale: float = 0.7,
        font_thickness: int = 2
    ):
        self.window_name = window_name
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX
        
    def create_window(self):
        """Create the display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def draw_rounded_rect(self, frame, pt1, pt2, color, thickness=-1, radius=10):
        """Draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        if thickness == -1:
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            cv2.rectangle(frame, pt1, pt2, color, thickness)
    
    def draw_confidence_bar(self, frame, x, y, width, height, confidence, color):
        """Draw a confidence progress bar."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS["bg_dark"], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS["text_dim"], 1)
        
        # Fill
        fill_width = int(width * confidence)
        if fill_width > 0:
            cv2.rectangle(frame, (x + 2, y + 2), (x + fill_width - 2, y + height - 2), color, -1)
        
        # Percentage text
        pct_text = f"{confidence:.1%}"
        text_size = cv2.getTextSize(pct_text, self.font, 0.5, 1)[0]
        text_x = x + (width - text_size[0]) // 2
        cv2.putText(frame, pct_text, (text_x, y + height - 5), self.font, 0.5, self.COLORS["text"], 1)
    
    def draw_stat_box(self, frame, x, y, label, value, color, width=120, height=50):
        """Draw a statistics box."""
        # Box background
        self.draw_rounded_rect(frame, (x, y), (x + width, y + height), self.COLORS["bg_panel"], -1, 5)
        
        # Label
        cv2.putText(frame, label, (x + 5, y + 15), self.font, 0.4, self.COLORS["text_dim"], 1)
        
        # Value
        cv2.putText(frame, str(value), (x + 5, y + 40), self.font_bold, 0.7, color, 2)
    
    def draw_prediction_with_stats(
        self,
        frame,
        label: str,
        confidence: float,
        fps: float = 0.0,
        top_predictions: list = None,
        stats: dict = None
    ):
        """
        Draw prediction results with live statistics on the frame.
        
        Args:
            frame: The video frame to draw on
            label: Predicted class label
            confidence: Prediction confidence
            fps: Current FPS
            top_predictions: List of top predictions
            stats: Statistics dictionary from StatisticsTracker
            
        Returns:
            Frame with annotations
        """
        height, width = frame.shape[:2]
        
        # Format the label
        display_label = label.replace("___", " - ").replace("_", " ")
        if len(display_label) > 30:
            display_label = display_label[:27] + "..."
        
        # Determine status color
        is_healthy = "healthy" in label.lower()
        status_color = self.COLORS["healthy"] if is_healthy else self.COLORS["disease"]
        status_text = "HEALTHY" if is_healthy else "DISEASE DETECTED"
        
        # Create overlay for semi-transparent panels
        overlay = frame.copy()
        
        # ===== TOP PANEL - Main Prediction =====
        panel_height = 100
        cv2.rectangle(overlay, (0, 0), (width, panel_height), self.COLORS["bg_dark"], -1)
        
        # Status indicator bar
        cv2.rectangle(overlay, (0, 0), (width, 8), status_color, -1)
        
        # ===== RIGHT PANEL - Statistics =====
        stats_width = 180
        if stats:
            cv2.rectangle(overlay, (width - stats_width, panel_height), 
                         (width, height), self.COLORS["bg_dark"], -1)
        
        # ===== BOTTOM PANEL - Top Predictions =====
        bottom_panel_height = 100
        cv2.rectangle(overlay, (0, height - bottom_panel_height), 
                     (width - stats_width if stats else width, height), 
                     self.COLORS["bg_dark"], -1)
        
        # Apply transparency
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # ===== DRAW TOP PANEL CONTENT =====
        # Status label (large)
        cv2.putText(frame, status_text, (15, 35), self.font_bold, 0.8, status_color, 2)
        
        # Prediction label (prominent)
        cv2.putText(frame, display_label, (15, 65), self.font_bold, 0.9, self.COLORS["text"], 2)
        
        # Confidence bar
        self.draw_confidence_bar(frame, 15, 75, 200, 18, confidence, status_color)
        
        # FPS indicator
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (width - 100, 30), self.font, 0.6, self.COLORS["info"], 1)
        
        # Frame counter
        if stats:
            frame_text = f"Frame: {stats.get('total_frames', 0)}"
            cv2.putText(frame, frame_text, (width - 100, 55), self.font, 0.5, self.COLORS["text_dim"], 1)
        
        # ===== DRAW STATISTICS PANEL =====
        if stats:
            stats_x = width - stats_width + 10
            stats_y = panel_height + 15
            
            # Title
            cv2.putText(frame, "LIVE STATISTICS", (stats_x, stats_y), 
                       self.font, 0.5, self.COLORS["accent"], 1)
            stats_y += 25
            
            # Separator line
            cv2.line(frame, (stats_x, stats_y), (width - 10, stats_y), self.COLORS["text_dim"], 1)
            stats_y += 15
            
            # Session time
            elapsed = stats.get('elapsed_time', 0)
            mins, secs = divmod(int(elapsed), 60)
            cv2.putText(frame, "Session Time", (stats_x, stats_y), self.font, 0.4, self.COLORS["text_dim"], 1)
            stats_y += 20
            cv2.putText(frame, f"{mins:02d}:{secs:02d}", (stats_x, stats_y), self.font_bold, 0.7, self.COLORS["text"], 2)
            stats_y += 30
            
            # Processed frames
            cv2.putText(frame, "Predictions", (stats_x, stats_y), self.font, 0.4, self.COLORS["text_dim"], 1)
            stats_y += 20
            cv2.putText(frame, str(stats.get('processed_frames', 0)), (stats_x, stats_y), 
                       self.font_bold, 0.7, self.COLORS["info"], 2)
            stats_y += 30
            
            # Disease rate
            cv2.putText(frame, "Disease Rate", (stats_x, stats_y), self.font, 0.4, self.COLORS["text_dim"], 1)
            stats_y += 20
            disease_pct = stats.get('disease_percentage', 0)
            disease_color = self.COLORS["disease"] if disease_pct > 50 else self.COLORS["healthy"]
            cv2.putText(frame, f"{disease_pct:.1f}%", (stats_x, stats_y), 
                       self.font_bold, 0.7, disease_color, 2)
            stats_y += 30
            
            # Average confidence
            cv2.putText(frame, "Avg Confidence", (stats_x, stats_y), self.font, 0.4, self.COLORS["text_dim"], 1)
            stats_y += 20
            avg_conf = stats.get('avg_confidence', 0)
            cv2.putText(frame, f"{avg_conf:.1%}", (stats_x, stats_y), 
                       self.font_bold, 0.7, self.COLORS["text"], 2)
            stats_y += 35
            
            # Top diseases header
            cv2.putText(frame, "TOP DETECTIONS", (stats_x, stats_y), 
                       self.font, 0.45, self.COLORS["warning"], 1)
            stats_y += 5
            cv2.line(frame, (stats_x, stats_y), (width - 10, stats_y), self.COLORS["text_dim"], 1)
            stats_y += 18
            
            # Top disease list
            top_diseases = stats.get('top_diseases', [])[:4]
            for disease_label, count in top_diseases:
                short_label = disease_label.replace("___", "-").replace("_", " ")
                if len(short_label) > 18:
                    short_label = short_label[:15] + "..."
                
                is_healthy_disease = "healthy" in disease_label.lower()
                label_color = self.COLORS["healthy"] if is_healthy_disease else self.COLORS["disease"]
                
                cv2.putText(frame, short_label, (stats_x, stats_y), self.font, 0.35, label_color, 1)
                cv2.putText(frame, str(count), (width - 35, stats_y), self.font, 0.4, self.COLORS["text"], 1)
                stats_y += 18
        
        # ===== DRAW BOTTOM PANEL - Top Predictions =====
        bottom_y = height - bottom_panel_height + 20
        cv2.putText(frame, "TOP PREDICTIONS", (15, bottom_y), self.font, 0.5, self.COLORS["accent"], 1)
        bottom_y += 5
        cv2.line(frame, (15, bottom_y), (250, bottom_y), self.COLORS["text_dim"], 1)
        bottom_y += 20
        
        if top_predictions:
            for i, (pred_label, pred_conf) in enumerate(top_predictions[:3]):
                pred_display = pred_label.replace("___", " - ").replace("_", " ")
                if len(pred_display) > 35:
                    pred_display = pred_display[:32] + "..."
                
                is_healthy_pred = "healthy" in pred_label.lower()
                pred_color = self.COLORS["healthy"] if is_healthy_pred else self.COLORS["disease"]
                
                # Rank number
                cv2.putText(frame, f"{i+1}.", (15, bottom_y), self.font, 0.5, self.COLORS["text_dim"], 1)
                
                # Label
                cv2.putText(frame, pred_display, (35, bottom_y), self.font, 0.5, pred_color, 1)
                
                # Confidence bar (mini)
                bar_x = 300
                bar_width = 100
                cv2.rectangle(frame, (bar_x, bottom_y - 12), (bar_x + bar_width, bottom_y + 2), 
                             self.COLORS["bg_panel"], -1)
                fill_w = int(bar_width * pred_conf)
                cv2.rectangle(frame, (bar_x, bottom_y - 12), (bar_x + fill_w, bottom_y + 2), 
                             pred_color, -1)
                cv2.putText(frame, f"{pred_conf:.1%}", (bar_x + bar_width + 5, bottom_y), 
                           self.font, 0.4, self.COLORS["text"], 1)
                
                bottom_y += 25
        
        # ===== BORDER =====
        cv2.rectangle(frame, (2, 2), (width-2, height-2), status_color, 3)
        
        return frame
    
    def draw_prediction(
        self,
        frame,
        label: str,
        confidence: float,
        fps: float = 0.0,
        top_predictions: list = None
    ):
        """Legacy method - calls draw_prediction_with_stats without stats."""
        return self.draw_prediction_with_stats(frame, label, confidence, fps, top_predictions, None)
    
    def show_frame(self, frame) -> int:
        """Display a frame and return the key pressed."""
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF
    
    def destroy_window(self):
        """Destroy the display window."""
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Window may not exist in headless mode
