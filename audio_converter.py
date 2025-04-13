import os
import subprocess
import logging
import datetime
import time
import traceback
import json
import threading
import queue
import uuid
from pathlib import Path


class AudioConverter:
    """
    Audio Converter library for extracting audio tracks from video files.
    This module focuses solely on converting video files to audio format,
    optimized for speech recognition and transcription purposes.
    """
    def __init__(self, config=None):
        """
        Initialize the Audio Converter with the provided configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with processing parameters.
                If None, default configuration will be used.
        """
        # Default configuration optimized for transcription
        self.config = {
            'output_dir': './extracted_audio',
            'output_format': 'mp3',          # Default format
            'audio_bitrate': '128k',         # Lower bitrate is fine for speech
            'audio_channels': 1,             # Mono is better for speech recognition
            'audio_sample_rate': 16000,      # 16kHz is standard for many speech recognition systems
            'ffmpeg_path': 'ffmpeg',         # Path to ffmpeg executable
            'log_level': logging.INFO,
            'normalize_audio': True,         # Apply audio normalization
            'remove_noise': False,           # Option to apply noise reduction
            'max_threads': 4,                # Maximum number of concurrent conversions
            'temp_dir': './temp'             # Temporary directory for processing
        }
        
        # Update with user-provided config
        if config:
            self.config.update(config)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Ensure output and temp directories exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['temp_dir'], exist_ok=True)
        
        # Set up conversion queue and workers
        self.conversion_queue = queue.Queue()
        self.workers = []
        self.active = False
        
        # Set up status tracking
        self.status = {
            'total_files': 0,
            'completed_files': 0,
            'failed_files': 0,
            'in_progress_files': 0,
            'total_duration': 0,
            'start_time': None,
            'end_time': None,
            'currently_processing': []
        }
        
        # Dictionary to store results of conversions
        self.results = {}
        
        # Check if ffmpeg is available
        self._check_ffmpeg()
        
        self.logger.info("Audio Converter initialized with configuration: %s", self.config)
    
    def _setup_logging(self):
        """
        Set up basic logging configuration.
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('audio_converter')
        logger.setLevel(self.config['log_level'])
        
        # Create console handler if no handlers exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Create file handler for persistent logs
            try:
                log_dir = os.path.join(self.config['output_dir'], 'logs')
                os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(os.path.join(log_dir, 'audio_conversion.log'))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not set up file logging: {e}")
        
        return logger
    
    def _check_ffmpeg(self):
        """
        Check if FFmpeg is available in the system path or specified location.
        """
        ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        try:
            result = subprocess.run(
                [ffmpeg_path, '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode == 0:
                self.logger.info("FFmpeg detected: %s", result.stdout.splitlines()[0])
            else:
                self.logger.warning("FFmpeg test returned non-zero code: %s", result.stderr)
        except Exception as e:
            self.logger.error("FFmpeg check failed: %s", e)
            self.logger.warning("Audio conversion will not work without FFmpeg")
    
    def convert_video_to_audio(self, video_path, output_path=None, format=None, bitrate=None, 
                               overwrite=False, additional_options=None, job_id=None):
        """
        Convert a video file to audio using FFmpeg.
        
        Args:
            video_path (str): Path to the video file
            output_path (str, optional): Path to save the extracted audio.
                If None, will be generated based on video path and output directory.
            format (str, optional): Audio format (mp3, wav, etc.). If None, use config value.
            bitrate (str, optional): Audio bitrate (e.g., '128k'). If None, use config value.
            overwrite (bool): Whether to overwrite if output file exists
            additional_options (list, optional): Additional FFmpeg options
            job_id (str, optional): Unique ID for this conversion job
            
        Returns:
            tuple: (success, output_path, error_message, job_id)
        """
        # Generate a job ID if none provided
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        try:
            # Validate video file exists
            if not os.path.exists(video_path):
                error = f"Video file does not exist: {video_path}"
                self.logger.error(error)
                return False, None, error, job_id
            
            # Set defaults from config
            if format is None:
                format = self.config.get('output_format', 'mp3')
            if bitrate is None:
                bitrate = self.config.get('audio_bitrate', '128k')
            
            # Generate output path if not provided
            if output_path is None:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(
                    self.config.get('output_dir', './extracted_audio'),
                    f"{video_name}.{format}"
                )
            
            # Check if output file already exists
            if os.path.exists(output_path) and not overwrite:
                self.logger.warning(f"Output file already exists: {output_path}")
                return False, None, "Output file already exists and overwrite=False", job_id
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build FFmpeg command
            ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
            command = [
                ffmpeg_path,
                '-y' if overwrite else '-n',  # Overwrite or not
                '-i', video_path,             # Input file
                '-vn',                        # Disable video
                '-acodec', self._get_audio_codec(format),  # Audio codec
                '-ab', bitrate,               # Audio bitrate
                '-ar', str(self.config.get('audio_sample_rate', 16000)),  # Sample rate
                '-ac', str(self.config.get('audio_channels', 1))  # Channels (mono for speech)
            ]
            
            # Add normalization if requested
            if self.config.get('normalize_audio', True):
                command.extend(['-af', 'loudnorm=I=-16:TP=-1.5:LRA=11'])
            
            # Add noise reduction if requested
            if self.config.get('remove_noise', False):
                if 'loudnorm' in command:
                    # Find the index of loudnorm and update it
                    for i, arg in enumerate(command):
                        if arg == '-af':
                            command[i+1] = 'afftdn=nf=-25,' + command[i+1]
                            break
                else:
                    command.extend(['-af', 'afftdn=nf=-25'])
            
            # Add any additional options
            if additional_options:
                command.extend(additional_options)
            
            # Add output path
            command.append(output_path)
            
            # Execute FFmpeg command
            self.logger.info("Converting video to audio: %s -> %s", video_path, output_path)
            self.logger.debug("FFmpeg command: %s", ' '.join(command))
            
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode == 0:
                self.logger.info("Audio conversion successful: %s", output_path)
                # Try to get audio duration
                duration = self._get_audio_duration(output_path)
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                
                result = {
                    'success': True, 
                    'output_path': output_path,
                    'error': None,
                    'duration': duration,
                    'file_size': file_size,
                    'format': format,
                    'bitrate': bitrate,
                    'job_id': job_id
                }
                
                self.results[job_id] = result
                return True, output_path, None, job_id
            else:
                error = process.stderr
                self.logger.error("FFmpeg error: %s", error)
                
                result = {
                    'success': False, 
                    'output_path': None,
                    'error': f"FFmpeg error: {error}",
                    'job_id': job_id
                }
                
                self.results[job_id] = result
                return False, None, f"FFmpeg error: {error}", job_id
                
        except Exception as e:
            error_message = f"Audio conversion failed: {str(e)}"
            self.logger.error(error_message)
            traceback.print_exc()
            
            result = {
                'success': False, 
                'output_path': None,
                'error': error_message,
                'job_id': job_id
            }
            
            self.results[job_id] = result
            return False, None, error_message, job_id
    
    def _get_audio_codec(self, format):
        """
        Get appropriate audio codec based on the output format.
        
        Args:
            format (str): Audio format (mp3, wav, etc.)
            
        Returns:
            str: FFmpeg audio codec parameter
        """
        format = format.lower()
        codec_map = {
            'mp3': 'libmp3lame',
            'aac': 'aac',
            'ogg': 'libvorbis',
            'flac': 'flac',
            'wav': 'pcm_s16le',
            'm4a': 'aac',
            'opus': 'libopus'
        }
        return codec_map.get(format, 'copy')  # Default to stream copy if unknown
    
    def _get_audio_duration(self, audio_path):
        """
        Try to get the duration of an audio file using FFmpeg.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            float: Duration in seconds, or None if could not be determined
        """
        ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        try:
            command = [
                ffmpeg_path,
                '-i', audio_path,
                '-f', 'null',
                '-'
            ]
            
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # FFmpeg outputs duration info to stderr
            stderr = process.stderr
            
            # Try to extract duration
            import re
            duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', stderr)
            if duration_match:
                hours, minutes, seconds, centiseconds = map(int, duration_match.groups())
                duration = hours * 3600 + minutes * 60 + seconds + centiseconds / 100
                return duration
            
            return None
            
        except Exception as e:
            self.logger.debug("Error getting audio duration: %s", e)
            return None
    
    def add_to_queue(self, video_path, output_path=None, format=None, bitrate=None, overwrite=False, additional_options=None):
        """
        Add a video file to the conversion queue.
        
        Args:
            video_path (str): Path to the video file
            output_path (str, optional): Path to save the extracted audio
            format (str, optional): Audio format (mp3, wav, etc.)
            bitrate (str, optional): Audio bitrate (e.g., '128k')
            overwrite (bool): Whether to overwrite if output file exists
            additional_options (list, optional): Additional FFmpeg options
            
        Returns:
            str: Job ID for tracking this conversion
        """
        job_id = str(uuid.uuid4())
        
        # Prepare job info
        job = {
            'video_path': video_path,
            'output_path': output_path,
            'format': format,
            'bitrate': bitrate,
            'overwrite': overwrite,
            'additional_options': additional_options,
            'job_id': job_id,
            'status': 'queued',
            'submitted_time': datetime.datetime.now().isoformat()
        }
        
        # Add to queue
        self.conversion_queue.put(job)
        self.status['total_files'] += 1
        
        self.logger.info(f"Added video to conversion queue: {video_path} (Job ID: {job_id})")
        return job_id
    
    def process_queue(self, max_threads=None):
        """
        Start processing the conversion queue with multiple worker threads.
        
        Args:
            max_threads (int, optional): Maximum number of threads to use.
                If None, use the value from config.
                
        Returns:
            bool: True if processing started, False if already running
        """
        if self.active:
            self.logger.warning("Queue processing is already active")
            return False
        
        if max_threads is None:
            max_threads = self.config.get('max_threads', 4)
        
        # Reset or initialize status
        if self.status['start_time'] is None:
            self.status['start_time'] = datetime.datetime.now()
            self.status['completed_files'] = 0
            self.status['failed_files'] = 0
            self.status['in_progress_files'] = 0
            self.status['currently_processing'] = []
        
        self.active = True
        
        # Create and start worker threads
        self.workers = []
        for i in range(max_threads):
            worker = threading.Thread(target=self._worker_thread, name=f"converter-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
        self.logger.info(f"Started queue processing with {max_threads} workers")
        return True
    
    def _worker_thread(self):
        """
        Worker thread function that processes jobs from the conversion queue.
        """
        while self.active:
            try:
                # Get job from queue with timeout to check active flag periodically
                try:
                    job = self.conversion_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Update status
                self.status['in_progress_files'] += 1
                self.status['currently_processing'].append(job['video_path'])
                
                self.logger.info(f"Starting conversion of {job['video_path']} (Job ID: {job['job_id']})")
                
                # Process the job
                success, output_path, error, job_id = self.convert_video_to_audio(
                    job['video_path'],
                    job['output_path'],
                    job['format'],
                    job['bitrate'],
                    job['overwrite'],
                    job['additional_options'],
                    job['job_id']
                )
                
                # Update status
                self.status['in_progress_files'] -= 1
                self.status['currently_processing'].remove(job['video_path'])
                
                if success:
                    self.status['completed_files'] += 1
                    # Add duration to total if available
                    result = self.results.get(job_id, {})
                    duration = result.get('duration')
                    if duration:
                        self.status['total_duration'] += duration
                else:
                    self.status['failed_files'] += 1
                
                # Mark task as done
                self.conversion_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in worker thread: {str(e)}")
                traceback.print_exc()
    
    def wait_for_completion(self, timeout=None):
        """
        Wait for all queued conversions to complete.

        Args:
            timeout (float, optional): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if all conversions completed, False if timeout occurred
        """
        if timeout is None:
            # Wait indefinitely
            self.conversion_queue.join()
            self.status['end_time'] = datetime.datetime.now()
            return True
        else:
            # Wait with timeout
            end_time = time.time() + timeout
            while time.time() < end_time:
                if self.conversion_queue.empty() and self.status['in_progress_files'] == 0:
                    self.status['end_time'] = datetime.datetime.now()
                    return True
                time.sleep(0.1)  # Small sleep to prevent CPU spinning
            return False
    
    def stop_processing(self):
        """
        Stop processing the conversion queue.
        
        Returns:
            int: Number of remaining items in queue
        """
        self.active = False
        
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        # Get count of remaining items
        remaining = self.conversion_queue.qsize()
        
        self.logger.info(f"Stopped queue processing. {remaining} items remaining in queue.")
        return remaining
    
    def get_status(self):
        """
        Get the current conversion status.
        
        Returns:
            dict: Status information
        """
        status = self.status.copy()
        
        # Calculate elapsed time
        if status['start_time']:
            end_time = status['end_time'] if status['end_time'] else datetime.datetime.now()
            elapsed = (end_time - status['start_time']).total_seconds()
            status['elapsed_seconds'] = elapsed
            
            # Calculate progress percentage
            if status['total_files'] > 0:
                status['progress_percent'] = (status['completed_files'] + status['failed_files']) / status['total_files'] * 100
            else:
                status['progress_percent'] = 0
                
            # Calculate conversion rate and estimated time remaining
            if elapsed > 0 and status['completed_files'] > 0:
                rate = status['completed_files'] / elapsed  # files per second
                status['conversion_rate'] = rate
                
                remaining_files = status['total_files'] - status['completed_files'] - status['failed_files']
                if remaining_files > 0 and rate > 0:
                    status['estimated_seconds_remaining'] = remaining_files / rate
                else:
                    status['estimated_seconds_remaining'] = 0
            else:
                status['conversion_rate'] = 0
                status['estimated_seconds_remaining'] = 0
        
        return status
    
    def get_result(self, job_id):
        """
        Get the result of a specific conversion job.
        
        Args:
            job_id (str): The job ID to look up
            
        Returns:
            dict: Result information for the job, or None if not found
        """
        return self.results.get(job_id)
    
    def get_all_results(self):
        """
        Get results for all conversion jobs.
        
        Returns:
            dict: Dictionary mapping job IDs to result information
        """
        return self.results
    
    def generate_report(self):
        """
        Generate a comprehensive report of all conversion activity.
        
        Returns:
            dict: Report data
        """
        status = self.get_status()
        
        # Count successes and failures
        successes = sum(1 for result in self.results.values() if result.get('success', False))
        failures = sum(1 for result in self.results.values() if not result.get('success', False))
        
        # Calculate total duration
        total_duration = sum(result.get('duration', 0) for result in self.results.values() if result.get('success', False))
        
        # Calculate total audio size
        total_size = sum(result.get('file_size', 0) for result in self.results.values() if result.get('success', False))
        
        report = {
            'start_time': self.status['start_time'].isoformat() if self.status['start_time'] else None,
            'end_time': self.status['end_time'].isoformat() if self.status['end_time'] else None,
            'elapsed_seconds': status.get('elapsed_seconds', 0),
            'total_files': self.status['total_files'],
            'completed_files': successes,
            'failed_files': failures,
            'total_audio_duration_seconds': total_duration,
            'total_audio_size_bytes': total_size,
            'output_format': self.config.get('output_format'),
            'audio_settings': {
                'bitrate': self.config.get('audio_bitrate'),
                'sample_rate': self.config.get('audio_sample_rate'),
                'channels': self.config.get('audio_channels'),
                'normalized': self.config.get('normalize_audio'),
                'noise_reduction': self.config.get('remove_noise')
            }
        }
        
        # Add human-readable summaries
        if report['elapsed_seconds'] > 0:
            minutes = report['elapsed_seconds'] / 60
            report['summary'] = (
                f"Processed {report['total_files']} files in {minutes:.1f} minutes. "
                f"Successfully converted {successes} files "
                f"({(successes/report['total_files']*100):.1f}% success rate). "
                f"Total audio duration: {total_duration/60:.1f} minutes."
            )
        
        return report
    
    def export_results_to_json(self, filename=None):
        """
        Export all conversion results to a JSON file.
        
        Args:
            filename (str, optional): Path to save the JSON file.
                If None, use a default name in the output directory.
                
        Returns:
            str: Path to the created JSON file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.config['output_dir'], f"conversion_results_{timestamp}.json")
        
        # Create report data
        report = self.generate_report()
        report['results'] = self.results
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Exported results to {filename}")
        return filename


# Example usage of the Audio Converter
def example_usage():
    """Example demonstrating the use of AudioConverter class"""
    
    # Configuration - optimized for transcription
    config = {
        'output_dir': './extracted_audio',
        'output_format': 'mp3',
        'audio_bitrate': '128k',
        'audio_channels': 1,        # Mono is better for speech recognition
        'audio_sample_rate': 16000, # 16kHz is standard for many speech recognition systems
        'normalize_audio': True,    # Apply audio normalization
        'max_threads': 2
    }
    
    # Initialize the converter
    converter = AudioConverter(config)
    
    try:
        # Method 1: Convert a single video file
        print("Converting a single video...")
        success, output_path, error, job_id = converter.convert_video_to_audio('sample_video.mp4')
        
        if success:
            print(f"Audio extracted to: {output_path}")
        else:
            print(f"Conversion failed: {error}")
        
        # Method 2: Process multiple videos with the queue
        print("\nProcessing multiple videos via queue...")
        # Add some videos to the queue
        job_ids = []
        for video_file in ['video1.mp4', 'video2.mkv', 'video3.avi']:
            job_id = converter.add_to_queue(video_file)
            job_ids.append(job_id)
        
        # Start processing with 2 worker threads
        converter.process_queue(max_threads=2)
        
        # Wait for completion (with a timeout of 60 seconds for this example)
        completed = converter.wait_for_completion(timeout=60)
        
        if completed:
            print("All conversions completed!")
        else:
            print("Timeout reached, not all conversions completed")
            # Stop processing
            converter.stop_processing()
        
        # Get results
        for job_id in job_ids:
            result = converter.get_result(job_id)
            if result and result.get('success'):
                print(f"Job {job_id} completed successfully: {result['output_path']}")
            else:
                print(f"Job {job_id} failed: {result.get('error')}")
        
        # Generate and display report
        report = converter.generate_report()
        print("\nConversion Report:")
        print(f"Total Files: {report['total_files']}")
        print(f"Completed: {report['completed_files']}")
        print(f"Failed: {report['failed_files']}")
        print(f"Total Duration: {report['total_audio_duration_seconds']/60:.2f} minutes")
        
        # Export results to JSON
        converter.export_results_to_json()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run the example if this file is executed directly
    example_usage()