#!/usr/bin/env python3
"""
Crop Disease Detection System - Main Application

This application provides real-time crop disease detection using a pretrained
deep learning model. It supports multiple input sources including webcam,
Raspberry Pi camera, video files, and image files.

Usage:
    python main.py                          # Use test video (default for development)
    python main.py --camera                 # Use webcam (production mode)
    python main.py --source video.mp4       # Use specific video file
    python main.py --image path/to/image    # Process single image or folder
    python main.py --picamera               # Use Raspberry Pi camera
    python main.py --help                   # Show help
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    DEFAULT_CLASS_LABELS,
    VIDEO_SETTINGS,
    DISPLAY_SETTINGS,
    LOGGING_CONFIG,
    BASE_DIR,
    TEST_VIDEO_PATH,
    RASPBERRY_PI_SETTINGS
)
from src.model_loader import CropDiseaseModel, create_model
from src.video_processor import VideoProcessor, VideoConfig, DisplayManager
from src.statistics_tracker import StatisticsTracker
from src.data_exporter import ExcelExporter, create_exporter


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log")
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crop Disease Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Use test video (development mode)
  python main.py --camera            Use webcam (production mode)
  python main.py --source video.mp4  Process a specific video file
  python main.py --image img.jpg     Process a single image
  python main.py --image data/images Process all images in a folder
  python main.py --picamera          Use Raspberry Pi camera
  python main.py --model custom.h5   Use a custom model file
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Video source: camera index (0, 1, ...) or path to video file"
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to image file or folder of images to process"
    )
    
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use webcam/camera for live video (production mode)"
    )
    
    parser.add_argument(
        "--picamera", "-p",
        action="store_true",
        help="Use Raspberry Pi camera (requires picamera2)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to custom model file (.h5 for TensorFlow, .pth for PyTorch)"
    )
    
    parser.add_argument(
        "--framework", "-f",
        choices=["tensorflow", "pytorch"],
        default="tensorflow",
        help="Deep learning framework to use (default: tensorflow)"
    )
    
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=VIDEO_SETTINGS["frame_width"],
        help=f"Frame width (default: {VIDEO_SETTINGS['frame_width']})"
    )
    
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=VIDEO_SETTINGS["frame_height"],
        help=f"Frame height (default: {VIDEO_SETTINGS['frame_height']})"
    )
    
    parser.add_argument(
        "--skip-frames", "-k",
        type=int,
        default=VIDEO_SETTINGS["skip_frames"],
        help=f"Process every nth frame (default: {VIDEO_SETTINGS['skip_frames']})"
    )
    
    parser.add_argument(
        "--confidence-threshold", "-c",
        type=float,
        default=DISPLAY_SETTINGS["confidence_threshold"],
        help=f"Minimum confidence threshold (default: {DISPLAY_SETTINGS['confidence_threshold']})"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display window (headless mode)"
    )
    
    parser.add_argument(
        "--save-output", "-o",
        type=str,
        default=None,
        help="Save output video to file"
    )
    
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Disable Excel export of detection results"
    )
    
    parser.add_argument(
        "--excel-dir",
        type=str,
        default=None,
        help="Directory to save Excel results (default: reports/)"
    )
    
    return parser.parse_args()


def process_images(args, model, logger):
    """
    Process image file(s) for disease detection.
    
    Args:
        args: Parsed command line arguments
        model: Loaded disease detection model
        logger: Logger instance
    """
    import cv2
    from pathlib import Path
    
    image_path = Path(args.image)
    
    # Determine if single image or directory
    if image_path.is_file():
        image_files = [image_path]
    elif image_path.is_dir():
        # Get all image files in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_files = [f for f in image_path.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions]
        if not image_files:
            logger.error(f"No image files found in: {image_path}")
            return
        logger.info(f"Found {len(image_files)} images in {image_path}")
    else:
        logger.error(f"Image path not found: {image_path}")
        return
    
    # Setup Excel exporter if enabled
    excel_exporter = None
    if not args.no_excel:
        excel_dir = args.excel_dir or str(BASE_DIR / "reports")
        excel_exporter = create_exporter(excel_dir, session_name="image_analysis")
        excel_exporter.set_session_info(
            video_source=str(image_path),
            model_name="HuggingFace MobileNetV2 (95.4% accuracy)",
            framework="pytorch",
            mode="image_analysis",
            total_images=len(image_files)
        )
    
    # Process each image
    results = []
    for idx, img_file in enumerate(image_files, 1):
        logger.info(f"Processing [{idx}/{len(image_files)}]: {img_file.name}")
        
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            logger.warning(f"Could not read image: {img_file}")
            continue
        
        # Run prediction
        label, confidence, top_preds = model.predict(image)
        
        # Check if above confidence threshold
        is_healthy = "healthy" in label.lower()
        status = "Healthy" if is_healthy else "Disease Detected"
        
        # Log result
        if confidence >= args.confidence_threshold:
            logger.info(f"  -> {label}: {confidence:.1%} ({status})")
        else:
            logger.info(f"  -> {label}: {confidence:.1%} (below threshold)")
        
        # Store result
        result = {
            "image": img_file.name,
            "prediction": label,
            "confidence": confidence,
            "status": status,
            "top_predictions": top_preds[:3] if top_preds else []
        }
        results.append(result)
        
        # Add to Excel
        if excel_exporter and confidence >= args.confidence_threshold:
            excel_exporter.add_detection(
                frame_number=idx,
                timestamp=0,
                prediction=label,
                confidence=confidence,
                top_predictions=top_preds,
                is_healthy=is_healthy,
                image_file=img_file.name
            )
        
        # Display image if not headless
        if not args.no_display:
            # Draw prediction on image
            display_text = f"{label}: {confidence:.1%}"
            color = (0, 255, 0) if is_healthy else (0, 0, 255)
            cv2.putText(image, display_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(image, f"[{idx}/{len(image_files)}] {img_file.name}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Crop Disease Detection - Image", image)
            key = cv2.waitKey(0 if len(image_files) == 1 else 2000)
            if key == ord('q'):
                break
    
    # Summary
    logger.info("=" * 50)
    logger.info("IMAGE ANALYSIS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Processed: {len(results)}")
    
    if results:
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        disease_count = sum(1 for r in results if r['status'] != 'Healthy')
        healthy_count = len(results) - disease_count
        
        logger.info(f"Average confidence: {avg_conf:.1%}")
        logger.info(f"Healthy: {healthy_count} ({healthy_count/len(results)*100:.1f}%)")
        logger.info(f"Disease detected: {disease_count} ({disease_count/len(results)*100:.1f}%)")
        
        # Count by prediction
        from collections import Counter
        pred_counts = Counter(r['prediction'] for r in results)
        logger.info("Predictions:")
        for pred, count in pred_counts.most_common(5):
            logger.info(f"  - {pred}: {count}")
    
    logger.info("=" * 50)
    
    # Save Excel
    if excel_exporter and excel_exporter.records:
        excel_path = excel_exporter.save()
        logger.info(f"Results saved to: {excel_path}")
    
    if not args.no_display:
        cv2.destroyAllWindows()
    
    return results


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("=" * 60)
    logger.info("Crop Disease Detection System")
    logger.info("=" * 60)
    
    # Check if image mode
    if args.image is not None:
        logger.info("Image analysis mode")
        logger.info(f"Input: {args.image}")
        
        # Load model first
        logger.info("Loading crop disease detection model...")
        try:
            model = create_model(
                model_path=args.model,
                class_labels=DEFAULT_CLASS_LABELS,
                framework=args.framework
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
        
        # Process images and exit
        process_images(args, model, logger)
        logger.info("Image analysis completed")
        return
    
    # Determine video source based on mode
    if args.source is not None:
        # Explicit source provided
        try:
            source = int(args.source)
            logger.info(f"Using camera index: {source}")
        except ValueError:
            source = args.source
            if not os.path.exists(source):
                logger.error(f"Video file not found: {source}")
                sys.exit(1)
            logger.info(f"Using video file: {source}")
    elif args.camera:
        # Camera mode (production)
        source = VIDEO_SETTINGS.get("camera_source", 0)
        logger.info(f"Production mode: Using camera index {source}")
    elif args.picamera:
        # Raspberry Pi camera - use optimized settings
        source = 0
        pi_res = RASPBERRY_PI_SETTINGS["resolution"]
        if args.width == 640:  # Default value, use Pi settings
            args.width = pi_res[0]
        if args.height == 480:  # Default value, use Pi settings
            args.height = pi_res[1]
        if args.skip_frames == 5:  # Default value, use Pi settings
            args.skip_frames = RASPBERRY_PI_SETTINGS["skip_frames"]
        logger.info(f"Using Raspberry Pi camera: {args.width}x{args.height}, skip_frames={args.skip_frames}")
    else:
        # Default: development mode with video file
        logger.info("Development mode: Using test video")
        if not TEST_VIDEO_PATH.exists():
            logger.error(f"Test video not found: {TEST_VIDEO_PATH}")
            logger.error("Please add a test video file to data/videos/")
            sys.exit(1)
        source = str(TEST_VIDEO_PATH)
        logger.info(f"Using video file: {source}")
    
    # Initialize model
    logger.info("Loading crop disease detection model...")
    try:
        model = create_model(
            model_path=args.model,
            class_labels=DEFAULT_CLASS_LABELS,
            framework=args.framework
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Configure video capture
    video_config = VideoConfig(
        source=source,
        width=args.width,
        height=args.height,
        use_picamera=args.picamera,
        skip_frames=args.skip_frames
    )
    
    # Initialize video processor
    video_processor = VideoProcessor(video_config)
    display_manager = DisplayManager(
        window_name=DISPLAY_SETTINGS["window_name"],
        font_scale=DISPLAY_SETTINGS["font_scale"],
        font_thickness=DISPLAY_SETTINGS["font_thickness"]
    )
    
    # Video writer for saving output
    video_writer = None
    if args.save_output:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save_output,
            fourcc,
            30.0,
            (args.width, args.height)
        )
        logger.info(f"Saving output to: {args.save_output}")
    
    # Initialize statistics tracker
    stats_tracker = StatisticsTracker(history_size=100)
    
    # Initialize Excel exporter
    excel_exporter = None
    if not args.no_excel:
        excel_dir = args.excel_dir or str(BASE_DIR / "reports")
        source_name = "camera" if args.camera or args.picamera else Path(str(source)).stem
        excel_exporter = create_exporter(excel_dir, session_name=source_name)
        excel_exporter.set_session_info(
            video_source=source,
            model_name="HuggingFace MobileNetV2 (95.4% accuracy)",
            framework="pytorch",
            skip_frames=args.skip_frames,
            confidence_threshold=args.confidence_threshold,
            resolution=f"{args.width}x{args.height}"
        )
        logger.info(f"Excel export enabled: {excel_exporter.filepath}")
    
    # Start video processing
    logger.info(f"Starting video capture from: {source}")
    logger.info("Press 'q' to quit, 's' to save screenshot, 'r' to reset stats")
    
    try:
        with video_processor:
            if not args.no_display:
                display_manager.create_window()
            
            current_prediction = None
            current_confidence = 0.0
            top_predictions = []
            current_stats = None
            
            for frame, frame_num, should_process in video_processor.get_frames():
                # Track frame
                stats_tracker.record_frame()
                
                # Run inference on selected frames
                if should_process:
                    try:
                        label, confidence, predictions = model.predict(frame)
                        
                        if confidence >= args.confidence_threshold:
                            current_prediction = label
                            current_confidence = confidence
                            top_predictions = predictions
                            
                            # Record prediction in stats
                            stats_tracker.record_prediction(label, confidence, frame_num)
                            
                            # Record to Excel exporter
                            if excel_exporter:
                                video_fps = video_processor.get_fps() or 30
                                timestamp_sec = frame_num / video_fps
                                is_healthy = "healthy" in label.lower()
                                excel_exporter.add_detection(
                                    frame_number=frame_num,
                                    timestamp=timestamp_sec,
                                    prediction=label,
                                    confidence=confidence,
                                    top_predictions=predictions,
                                    is_healthy=is_healthy
                                )
                            
                            logger.debug(
                                f"Frame {frame_num}: {label} ({confidence:.1%})"
                            )
                    except Exception as e:
                        logger.warning(f"Prediction error: {e}")
                
                # Get current statistics
                current_stats = stats_tracker.get_current_stats()
                
                # Draw results on frame with statistics
                if current_prediction:
                    display_frame = display_manager.draw_prediction_with_stats(
                        frame.copy(),
                        current_prediction,
                        current_confidence,
                        video_processor.get_fps(),
                        top_predictions,
                        current_stats
                    )
                else:
                    display_frame = frame.copy()
                    import cv2
                    cv2.putText(
                        display_frame,
                        "Initializing model...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 255),
                        2
                    )
                    cv2.putText(
                        display_frame,
                        "Please wait for first prediction",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (180, 180, 180),
                        1
                    )
                
                # Save frame to output video
                if video_writer:
                    video_writer.write(display_frame)
                
                # Display frame
                if not args.no_display:
                    key = display_manager.show_frame(display_frame)
                    
                    # Handle key presses
                    if key == ord('q'):
                        logger.info("Quit requested by user")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        import cv2
                        screenshot_dir = BASE_DIR / "screenshots"
                        screenshot_dir.mkdir(exist_ok=True)
                        screenshot_path = screenshot_dir / f"screenshot_{frame_num}.jpg"
                        cv2.imwrite(str(screenshot_path), display_frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                    elif key == ord('r'):
                        # Reset statistics
                        stats_tracker.reset()
                        logger.info("Statistics reset")
            
            # Final statistics summary
            final_stats = stats_tracker.get_current_stats()
            logger.info("=" * 50)
            logger.info("SESSION SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total frames: {final_stats['total_frames']}")
            logger.info(f"Predictions made: {final_stats['processed_frames']}")
            logger.info(f"Average confidence: {final_stats['avg_confidence']:.1%}")
            logger.info(f"Disease rate: {final_stats['disease_percentage']:.1f}%")
            logger.info(f"Healthy rate: {final_stats['healthy_percentage']:.1f}%")
            if final_stats['top_diseases']:
                logger.info("Top detections:")
                for disease, count in final_stats['top_diseases'][:5]:
                    logger.info(f"  - {disease}: {count}")
            logger.info("=" * 50)
            
            # Save Excel report
            if excel_exporter and excel_exporter.records:
                excel_path = excel_exporter.save()
                logger.info(f"Detection results saved to: {excel_path}")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Save Excel if we have any records (even on error/interrupt)
        if excel_exporter and excel_exporter.records:
            try:
                excel_path = excel_exporter.save()
                logger.info(f"Detection results saved to: {excel_path}")
            except Exception as save_err:
                logger.error(f"Failed to save Excel: {save_err}")
        
        # Cleanup
        if video_writer:
            video_writer.release()
        display_manager.destroy_window()
        logger.info("Application terminated")


if __name__ == "__main__":
    main()
