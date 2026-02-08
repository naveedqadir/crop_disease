"""
Statistics Tracker Module

Tracks and manages prediction statistics for real-time display
during crop disease detection.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import threading


@dataclass
class PredictionRecord:
    """Record of a single prediction."""
    timestamp: float
    label: str
    confidence: float
    frame_number: int


@dataclass 
class SessionStats:
    """Statistics for the current session."""
    start_time: float = field(default_factory=time.time)
    total_frames: int = 0
    processed_frames: int = 0
    predictions: List[PredictionRecord] = field(default_factory=list)
    disease_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    healthy_count: int = 0
    disease_count: int = 0
    confidence_sum: float = 0.0
    
    # Recent predictions for trend analysis
    recent_predictions: deque = field(default_factory=lambda: deque(maxlen=30))


class StatisticsTracker:
    """
    Tracks prediction statistics in real-time.
    Thread-safe for use in video processing loops.
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize the statistics tracker.
        
        Args:
            history_size: Number of recent predictions to keep
        """
        self.history_size = history_size
        self.stats = SessionStats()
        self._lock = threading.Lock()
        
        # Moving averages
        self.confidence_history = deque(maxlen=history_size)
        self.prediction_times = deque(maxlen=30)  # For FPS calculation
        
    def record_frame(self):
        """Record that a frame was processed."""
        with self._lock:
            self.stats.total_frames += 1
    
    def record_prediction(
        self, 
        label: str, 
        confidence: float, 
        frame_number: int
    ):
        """
        Record a prediction result.
        
        Args:
            label: Predicted class label
            confidence: Prediction confidence (0-1)
            frame_number: Current frame number
        """
        with self._lock:
            now = time.time()
            
            record = PredictionRecord(
                timestamp=now,
                label=label,
                confidence=confidence,
                frame_number=frame_number
            )
            
            self.stats.predictions.append(record)
            self.stats.processed_frames += 1
            self.stats.recent_predictions.append(record)
            
            # Update counts
            self.stats.disease_counts[label] += 1
            
            if "healthy" in label.lower():
                self.stats.healthy_count += 1
            else:
                self.stats.disease_count += 1
            
            # Update confidence tracking
            self.stats.confidence_sum += confidence
            self.confidence_history.append(confidence)
            self.prediction_times.append(now)
    
    def get_current_stats(self) -> Dict:
        """Get current statistics summary."""
        with self._lock:
            elapsed = time.time() - self.stats.start_time
            
            # Calculate prediction rate
            if len(self.prediction_times) > 1:
                time_span = self.prediction_times[-1] - self.prediction_times[0]
                pred_rate = len(self.prediction_times) / time_span if time_span > 0 else 0
            else:
                pred_rate = 0
            
            # Average confidence
            avg_conf = (
                self.stats.confidence_sum / self.stats.processed_frames
                if self.stats.processed_frames > 0 else 0
            )
            
            # Recent average confidence
            recent_conf = (
                sum(self.confidence_history) / len(self.confidence_history)
                if self.confidence_history else 0
            )
            
            # Top diseases
            sorted_diseases = sorted(
                self.stats.disease_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Disease percentage
            total_preds = self.stats.processed_frames
            disease_pct = (
                self.stats.disease_count / total_preds * 100
                if total_preds > 0 else 0
            )
            healthy_pct = (
                self.stats.healthy_count / total_preds * 100
                if total_preds > 0 else 0
            )
            
            return {
                "elapsed_time": elapsed,
                "total_frames": self.stats.total_frames,
                "processed_frames": self.stats.processed_frames,
                "prediction_rate": pred_rate,
                "avg_confidence": avg_conf,
                "recent_confidence": recent_conf,
                "disease_counts": dict(self.stats.disease_counts),
                "top_diseases": sorted_diseases[:5],
                "disease_percentage": disease_pct,
                "healthy_percentage": healthy_pct,
                "healthy_count": self.stats.healthy_count,
                "disease_count": self.stats.disease_count,
            }
    
    def get_recent_predictions(self, n: int = 5) -> List[PredictionRecord]:
        """Get the n most recent predictions."""
        with self._lock:
            recent = list(self.stats.recent_predictions)[-n:]
            return recent
    
    def get_dominant_prediction(self, window: int = 10) -> Tuple[str, float]:
        """
        Get the most common prediction in the recent window.
        
        Args:
            window: Number of recent predictions to consider
            
        Returns:
            Tuple of (most_common_label, percentage)
        """
        with self._lock:
            recent = list(self.stats.recent_predictions)[-window:]
            
            if not recent:
                return "Unknown", 0.0
            
            counts = defaultdict(int)
            for pred in recent:
                counts[pred.label] += 1
            
            dominant = max(counts.items(), key=lambda x: x[1])
            percentage = dominant[1] / len(recent)
            
            return dominant[0], percentage
    
    def get_confidence_trend(self) -> str:
        """Get the trend of confidence (increasing, stable, decreasing)."""
        with self._lock:
            if len(self.confidence_history) < 10:
                return "stable"
            
            recent = list(self.confidence_history)
            first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            
            diff = second_half - first_half
            
            if diff > 0.05:
                return "increasing"
            elif diff < -0.05:
                return "decreasing"
            else:
                return "stable"
    
    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self.stats = SessionStats()
            self.confidence_history.clear()
            self.prediction_times.clear()
    
    def format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
