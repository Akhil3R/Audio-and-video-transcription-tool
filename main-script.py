#!/usr/bin/env python3
"""
Video Transcription Tool

This script scans a directory for video files, converts them to audio,
and then transcribes the audio to text files. Transcripts are saved in the same 
directory as the original video files.
"""

import os
import sys
import logging
import time
import json
import hashlib
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

# Fix SSL certificate issues
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Import our modules
from video_processor import VideoProcessor
from audio_converter import AudioConverter
from audio_transcriber import AudioTranscriber

# Set up logging with proper path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "transcription_tool.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger("transcription_tool")

# CONFIGURATION SECTION
# Edit these settings to match your needs
CONFIG = {
    # Directory to scan for video files (EDIT THIS)
    'input_dir': '/Volumes/Database/EMC ',  # Note the space at the end
    
    # Directory to store temp audio files
    'audio_dir': './temp_audio',
    
    # Audio format for extracted audio (mp3, wav, ogg)
    'audio_format': 'mp3',
    
    # Whisper model size (tiny, base, small, medium, large)
    'whisper_model': 'base',
    
    # Recursively scan subdirectories
    'recursive': True,
    
    # Force retranscription of already processed files
    'force_retranscribe': False,
    
    # Apple M2 optimization - use more threads for parallel processing
    # The M2 has 8 cores, but we'll leave some for system processes
    'max_threads': min(6, multiprocessing.cpu_count()),
    
    # Directory for storing processing logs and database
    'tracking_dir': './tracking'
}

class ProcessedFilesTracker:
    """Track processed files to avoid reprocessing on restart."""
    
    def __init__(self, output_dir: str, force_retranscribe: bool = False):
        """Initialize the tracker."""
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "processed_files.json")
        self.processed_files = self._load_log()
        self.force_retranscribe = force_retranscribe
        
        if force_retranscribe:
            self.set_force_retranscribe(True)
        
    def _load_log(self) -> Dict[str, Dict[str, Any]]:
        """Load the log file if it exists."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error parsing {self.log_file}, starting with empty log")
                return {}
        return {}
    
    def _save_log(self):
        """Save the log to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    def is_processed(self, file_path: str) -> bool:
        """Check if a file has already been processed."""
        if self.force_retranscribe:
            return False
            
        # Generate expected transcript path
        video_dir = os.path.dirname(file_path)
        video_name = os.path.splitext(os.path.basename(file_path))[0]
        transcript_path = os.path.join(video_dir, f"{video_name}_transcribe.txt")
        
        # If transcript exists, consider it processed
        if os.path.exists(transcript_path):
            logger.debug(f"Found existing transcript: {transcript_path}")
            return True
            
        # Otherwise check the log
        file_hash = self._get_file_hash(file_path)
        
        # Check if file exists in log and has matching hash
        if file_path in self.processed_files:
            # If force_retranscribe is enabled, we always return False
            if self.processed_files[file_path].get('force_retranscribe', False):
                return False
            
            # If hash matches, file is processed
            if self.processed_files[file_path].get('hash') == file_hash:
                return True
                
            # If hash doesn't match, file has changed
            logger.info(f"File {file_path} changed since last processing (hash mismatch)")
            return False
            
        return False
    
    def mark_processed(self, file_path: str, transcript_path: str, audio_path: str):
        """Mark a file as processed."""
        file_hash = self._get_file_hash(file_path)
        
        self.processed_files[file_path] = {
            'hash': file_hash,
            'transcript_path': transcript_path,
            'audio_path': audio_path,
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self._save_log()
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get a hash of the file to detect changes."""
        # For very large files, just hash a portion to save time
        max_bytes = 1024 * 1024  # 1MB sample size
        
        file_size = os.path.getsize(file_path)
        
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                # For large files, read the first, middle and last parts
                if file_size > max_bytes * 3:
                    # First part
                    hasher.update(f.read(max_bytes))
                    
                    # Middle part
                    f.seek(file_size // 2)
                    hasher.update(f.read(max_bytes))
                    
                    # Last part
                    f.seek(max(0, file_size - max_bytes))
                    hasher.update(f.read(max_bytes))
                else:
                    # For smaller files, read the whole file
                    hasher.update(f.read())
                    
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error getting hash for {file_path}: {e}")
            # Return timestamp as fallback
            return str(os.path.getmtime(file_path))
    
    def get_processed_files_count(self) -> int:
        """Get the number of processed files."""
        return len(self.processed_files)
    
    def set_force_retranscribe(self, force_retranscribe: bool):
        """Set force retranscribe flag for all files."""
        for file_path in self.processed_files:
            self.processed_files[file_path]['force_retranscribe'] = force_retranscribe
        self._save_log()

def init_directories(config):
    """Initialize directories."""
    # Create tracking directory for logs and database
    os.makedirs(config['tracking_dir'], exist_ok=True)
    
    # Create audio temp directory
    os.makedirs(config['audio_dir'], exist_ok=True)
    
    return config

def scan_for_videos(input_dir: str, recursive: bool = True) -> List[str]:
    """
    Scan the input directory for video files.
    
    Args:
        input_dir: Directory to scan
        recursive: Whether to scan recursively
        
    Returns:
        List of paths to video files
    """
    logger.info(f"Scanning for video files in {input_dir}")
    
    # Initialize VideoProcessor with our configuration
    video_processor_config = {
        'log_dir': os.path.join(CONFIG['tracking_dir'], 'logs'),
        'log_level': logging.INFO,
        'threads': CONFIG['max_threads'],  # Use more threads for scanning on M2
        # Add .ts files to supported formats
        'video_extensions': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', 
                           '.m4v', '.mpg', '.mpeg', '.3gp', '.ts', '.mts', '.m2ts']
    }
    
    processor = VideoProcessor(video_processor_config)
    
    try:
        # Scan for video files
        video_files = processor.scan_directory(input_dir, recursive=recursive)
        logger.info(f"Found {len(video_files)} video files")
        return video_files
    except Exception as e:
        logger.error(f"Error scanning for videos: {e}")
        return []
    finally:
        processor.close()

def convert_videos_to_audio(
    video_files: List[str], 
    output_dir: str, 
    audio_format: str = "mp3", 
    max_threads: int = 4
) -> Dict[str, str]:
    """
    Convert video files to audio. Optimized for Apple M2 processor.
    
    Args:
        video_files: List of video file paths
        output_dir: Directory to save audio files
        audio_format: Audio format (mp3, wav, etc.)
        max_threads: Maximum number of concurrent conversions
        
    Returns:
        Dictionary mapping video paths to audio paths
    """
    logger.info(f"Converting {len(video_files)} video files to audio using {max_threads} threads")
    
    # Initialize AudioConverter with our configuration
    audio_converter_config = {
        'output_dir': output_dir,
        'output_format': audio_format,
        'audio_bitrate': '128k',
        'audio_channels': 1,        # Mono is better for speech recognition
        'audio_sample_rate': 16000, # 16kHz is standard for many speech recognition systems
        'normalize_audio': True,    # Apply audio normalization
        'max_threads': max_threads
    }
    
    converter = AudioConverter(audio_converter_config)
    
    # Dictionary to store video-to-audio mappings
    video_to_audio = {}
    
    try:
        # Add all videos to the queue
        job_ids = []
        for video_file in video_files:
            # Generate a unique name for the temp audio file
            video_path = Path(video_file)
            video_name = video_path.stem
            # Use a hashed name to avoid conflicts with videos that have the same filename in different directories
            file_hash = hashlib.md5(video_file.encode()).hexdigest()[:8]
            audio_filename = f"{video_name}_{file_hash}.{audio_format}"
            output_path = os.path.join(output_dir, audio_filename)
            
            # Skip if file already exists (optimizes processing)
            if os.path.exists(output_path):
                logger.info(f"Audio file already exists, skipping conversion: {output_path}")
                video_to_audio[video_file] = output_path
                continue
            
            # Add to queue
            job_id = converter.add_to_queue(
                video_file, 
                output_path=output_path,
                format=audio_format,
                overwrite=True
            )
            job_ids.append(job_id)
            
            # Store the mapping
            video_to_audio[video_file] = output_path
        
        if job_ids:
            # Start processing with parallel threads for M2 optimization
            converter.process_queue(max_threads=max_threads)
            
            # Wait for completion
            logger.info("Waiting for audio conversions to complete...")
            try:
                converter.wait_for_completion()
            except Exception as e:
                logger.error(f"Error waiting for audio conversions to complete: {e}")
            
            # Get results
            results = converter.get_all_results()
            
            # Check for failures and log them
            success_count = 0
            failure_count = 0
            
            for job_id, result in results.items():
                if result.get('success'):
                    success_count += 1
                    logger.info(f"Successfully converted: {result.get('output_path')}")
                else:
                    failure_count += 1
                    video_path = next((v for v, a in video_to_audio.items() 
                                      if a == result.get('output_path')), "Unknown")
                    logger.error(f"Failed to convert {video_path}: {result.get('error')}")
                    
                    # Remove failed conversions from the mapping
                    if video_path in video_to_audio:
                        del video_to_audio[video_path]
            
            logger.info(f"Audio conversion completed. Success: {success_count}, Failed: {failure_count}")
            
            # Generate report
            report = converter.generate_report()
            logger.info(f"Conversion summary: {report.get('summary', '')}")
        else:
            logger.info("No new audio conversions needed")
        
        return video_to_audio
        
    except Exception as e:
        logger.error(f"Error during audio conversion: {e}")
        return video_to_audio

def transcribe_audio_files(
    audio_files: List[str], 
    video_to_audio_map: Dict[str, str],
    model_size: str = "base"
) -> Dict[str, str]:
    """
    Transcribe audio files to text and save transcripts in the same directory as original videos.
    
    Args:
        audio_files: List of audio file paths
        video_to_audio_map: Dictionary mapping video paths to audio paths (reversed from convert_videos_to_audio output)
        model_size: Whisper model size
        
    Returns:
        Dictionary mapping audio paths to transcript paths
    """
    logger.info(f"Transcribing {len(audio_files)} audio files using Whisper {model_size} model")
    
    # Reverse the video_to_audio map to get audio_to_video
    audio_to_video = {audio_path: video_path for video_path, audio_path in video_to_audio_map.items()}
    
    # Initialize AudioTranscriber with temp configuration
    # We'll modify the output path for each file individually
    transcriber_config = {
        'output_dir': CONFIG['tracking_dir'],  # This will be overridden for each file
        'model_size': model_size,
        'language': None,  # Auto-detect language
        'device': 'cpu',   # Use CPU for transcription
        'return_timestamps': True
    }
    
    try:
        transcriber = AudioTranscriber(transcriber_config)
    except Exception as e:
        logger.error(f"Failed to initialize transcriber: {e}")
        logger.error("Make sure you have installed OpenAI Whisper with: pip install openai-whisper")
        return {}
    
    # Dictionary to store audio-to-transcript mappings
    audio_to_transcript = {}
    
    try:
        # Process each audio file
        for audio_file in audio_files:
            try:
                # Find the original video path for this audio
                if audio_file not in audio_to_video:
                    logger.error(f"Could not find corresponding video for audio file: {audio_file}")
                    continue
                
                video_path = audio_to_video[audio_file]
                video_dir = os.path.dirname(video_path)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                
                # Generate output path in the same directory as the video
                output_path = os.path.join(video_dir, f"{video_name}_transcribe.txt")
                
                # Skip if transcript already exists
                if os.path.exists(output_path) and not CONFIG['force_retranscribe']:
                    logger.info(f"Transcript already exists, skipping: {output_path}")
                    audio_to_transcript[audio_file] = output_path
                    continue
                
                logger.info(f"Transcribing {audio_file} to {output_path}")
                
                # Transcribe the audio
                start_time = time.time()
                result = transcriber.transcribe_audio(audio_file, output_path=output_path)
                
                # Calculate transcription time
                elapsed_time = time.time() - start_time
                logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
                
                # Store the mapping
                audio_to_transcript[audio_file] = output_path
                
                # Export to other formats in same directory as video
                export_base_path = os.path.join(video_dir, f"{video_name}_transcribe")
                export_paths = transcriber.export_transcript_formats(result, export_base_path)
                
                logger.info(f"Exported transcripts to formats: {', '.join(export_paths.keys())}")
                
            except Exception as e:
                logger.error(f"Error transcribing {audio_file}: {e}")
                continue
        
        logger.info(f"Transcription completed. Processed {len(audio_to_transcript)} out of {len(audio_files)} files")
        return audio_to_transcript
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return audio_to_transcript
        
def cleanup_temp_audio(audio_files):
    """Clean up temporary audio files."""
    logger.info(f"Cleaning up {len(audio_files)} temporary audio files")
    removed = 0
    for audio_file in audio_files:
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
                removed += 1
        except Exception as e:
            logger.error(f"Error removing temporary audio file {audio_file}: {e}")
    
    logger.info(f"Removed {removed} temporary audio files")

def process_videos():
    """Process videos from scan to transcription."""
    # Initialize directories
    config = init_directories(CONFIG)
    
    # Initialize the processed files tracker
    tracker = ProcessedFilesTracker(config['tracking_dir'], config['force_retranscribe'])
    
    # If force retranscribe is enabled, log it
    if config['force_retranscribe']:
        logger.info("Force retranscribe enabled - will reprocess all files")
    
    # Step 1: Scan for video files
    all_video_files = scan_for_videos(config['input_dir'], config['recursive'])
    
    if not all_video_files:
        logger.error("No video files found. Exiting.")
        return False
    
    # Filter out already processed files
    video_files = []
    skipped_files = []
    
    for video_path in all_video_files:
        if not config['force_retranscribe'] and tracker.is_processed(video_path):
            logger.info(f"Skipping already processed file: {video_path}")
            skipped_files.append(video_path)
        else:
            video_files.append(video_path)
    
    # Log statistics
    logger.info(f"Found {len(all_video_files)} video files total")
    logger.info(f"Skipping {len(skipped_files)} already processed files")
    logger.info(f"Processing {len(video_files)} new or changed files")
    
    if not video_files:
        logger.info("No new files to process. Exiting.")
        return True
    
    # Step 2: Convert videos to audio (temporary files)
    video_to_audio = convert_videos_to_audio(
        video_files, 
        config['audio_dir'], 
        config['audio_format'], 
        config['max_threads']
    )
    
    if not video_to_audio:
        logger.error("No audio files were created. Exiting.")
        return False
    
    # Get the list of successfully converted audio files
    audio_files = list(video_to_audio.values())
    
    # Step 3: Transcribe audio files and save transcripts beside original videos
    audio_to_transcript = transcribe_audio_files(
        audio_files,
        video_to_audio,
        config['whisper_model']
    )
    
    if not audio_to_transcript:
        logger.error("No transcripts were created.")
        # Clean up temporary audio files
        cleanup_temp_audio(audio_files)
        return False
    
    # Step 4: Mark files as processed and generate final report
    successful_videos = []
    for video_path, audio_path in video_to_audio.items():
        if audio_path in audio_to_transcript:
            transcript_path = audio_to_transcript[audio_path]
            
            # Mark as processed in the tracker
            tracker.mark_processed(video_path, transcript_path, audio_path)
            
            video_name = os.path.basename(video_path)
            successful_videos.append({
                'video': video_name,
                'video_dir': os.path.dirname(video_path),
                'transcript': os.path.basename(transcript_path)
            })
    
    # Clean up temporary audio files
    cleanup_temp_audio(audio_files)
    
    # Add skipped files to successful count if they were previously processed
    total_successful = len(successful_videos) + len(skipped_files)
    logger.info(f"Successfully processed {len(successful_videos)} new videos")
    logger.info(f"Total successful videos (including previously processed): {total_successful}")
    
    # Generate report including both new and previously processed files
    if successful_videos or skipped_files:
        # Write a summary file
        summary_path = os.path.join(config['tracking_dir'], "processing_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Video Transcription Summary\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total videos tracked: {total_successful}\n")
            f.write(f"New videos processed this run: {len(successful_videos)}\n")
            f.write(f"Previously processed videos: {len(skipped_files)}\n\n")
            
            if successful_videos:
                f.write("Newly processed videos:\n")
                for i, item in enumerate(successful_videos, 1):
                    f.write(f"{i}. Video: {item['video']}\n")
                    f.write(f"   Directory: {item['video_dir']}\n")
                    f.write(f"   Transcript: {item['transcript']}\n\n")
            
            if skipped_files:
                f.write("\nSkipped previously processed videos:\n")
                for i, video_path in enumerate(skipped_files, 1):
                    f.write(f"{i}. {video_path}\n")
        
        logger.info(f"Summary written to {summary_path}")
    
    return True

def main():
    """Main entry point."""
    start_time = time.time()
    
    try:
        # Process videos using the hardcoded configuration
        logger.info(f"Starting video transcription with configuration:")
        logger.info(f"  Input directory: {CONFIG['input_dir']}")
        logger.info(f"  Temp audio directory: {CONFIG['audio_dir']}")
        logger.info(f"  Whisper model: {CONFIG['whisper_model']}")
        logger.info(f"  Max threads: {CONFIG['max_threads']} (optimized for Apple M2)")
        logger.info(f"  Transcripts will be saved in the same directories as their source videos")
        
        success = process_videos()
        
        # Calculate and log total processing time
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        logger.info(f"Total processing time: {int(minutes)} minutes, {seconds:.2f} seconds")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())