import os
import re
import logging
import logging.handlers
import sqlite3
import time
import datetime
import hashlib
from collections import deque
import threading
from pathlib import Path
import queue
import shutil
import tempfile
import traceback
import sys
import struct

class VideoProcessor:
    """
    Video Processor library for directory scanning, file tracking, and processing management.
    This module handles directory scanning, metadata extraction, logging, database tracking,
    and queue management for video processing operations. Supports common video formats
    including MP4, AVI, MOV, WMV, FLV, MKV, WEBM, and more.
    """
    def __init__(self, config=None):
        """
        Initialize the Video Processor with the provided configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with processing parameters.
                If None, default configuration will be used.
        """
        # Default configuration
        self.config = {
            'log_dir': './logs',
            'log_level': logging.INFO,
            'log_retention': 30,  # days
            'db_path': './video_tracking.db',
            'batch_size': 10,
            'timeout': 300,  # seconds
            'max_recursion_depth': None,
            'max_file_size': 2 * 1024 * 1024 * 1024,  # 2 GB
            'exclusion_patterns': [],
            'threads': 1,
            'temp_dir': './temp',
            'video_extensions': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', 
                               '.m4v', '.mpg', '.mpeg', '.3gp', '.ts', '.mts', '.m2ts']
        }
        
        # Update with user-provided config
        if config:
            self.config.update(config)
        
        # Ensure required directories exist
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs(self.config['temp_dir'], exist_ok=True)
        
        # Initialize components
        self.logger = self.initialize_logging(self.config['log_dir'], self.config['log_level'])
        self.db_connection = self.initialize_database(self.config['db_path'])
        self.processing_queue = queue.PriorityQueue()
        
        # Set up status tracking dictionary
        self.processing_status = {
            'running': False,
            'total_files': 0,
            'processed_files': 0,
            'success_count': 0,
            'failure_count': 0,
            'current_file': None,
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info("Video Processor initialized with configuration: %s", self.config)
    
    def initialize_logging(self, log_dir, log_level):
        """
        Set up the logging system with file and console handlers.
        
        Args:
            log_dir (str): Directory to store log files
            log_level (int): Logging level (e.g., logging.INFO)
            
        Returns:
            logging.Logger: Configured logger
        """
        # Create logger
        logger = logging.getLogger('video_processor')
        logger.setLevel(log_level)
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        
        # Create file handlers for different log categories
        # Processing log - general processing information
        process_handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, 'processing.log'),
            when='midnight',
            backupCount=self.config['log_retention']
        )
        process_handler.setLevel(log_level)
        process_handler.setFormatter(file_formatter)
        
        # Error log - only error and critical messages
        error_handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, 'errors.log'),
            when='midnight',
            backupCount=self.config['log_retention']
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # System log - only system-level messages
        system_filter = logging.Filter('video_processor.system')
        system_handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, 'system.log'),
            when='midnight',
            backupCount=self.config['log_retention']
        )
        system_handler.setLevel(log_level)
        system_handler.setFormatter(file_formatter)
        system_handler.addFilter(system_filter)
        
        # Performance log - performance metrics
        perf_handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, 'performance.log'),
            when='midnight',
            backupCount=self.config['log_retention']
        )
        perf_handler.setLevel(log_level)
        perf_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(process_handler)
        logger.addHandler(error_handler)
        logger.addHandler(system_handler)
        logger.addHandler(perf_handler)
        
        logger.info("Logging system initialized at %s with level %s", log_dir, 
                   logging.getLevelName(log_level))
        
        return logger
    
    def initialize_database(self, db_path):
        """
        Create database tables if they don't exist.
        
        Args:
            db_path (str): Path to the SQLite database file
            
        Returns:
            sqlite3.Connection: Database connection
        """
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create files table to track video files
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_files (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT UNIQUE,
                    file_hash TEXT,
                    file_size INTEGER,
                    creation_date TEXT,
                    modification_date TEXT,
                    format TEXT,
                    duration REAL,
                    resolution TEXT,
                    status TEXT DEFAULT 'pending',
                    last_processed TEXT,
                    error_message TEXT,
                    processing_time REAL
                )
            ''')
            
            # Create processing history table for tracking multiple processing attempts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    processed_date TEXT,
                    status TEXT,
                    processing_time REAL,
                    error_message TEXT,
                    FOREIGN KEY (file_id) REFERENCES video_files (id)
                )
            ''')
            
            # Create processing jobs table for overall job tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    total_files INTEGER,
                    successful_files INTEGER,
                    failed_files INTEGER,
                    job_status TEXT
                )
            ''')
            
            conn.commit()
            self.logger.info("Database initialized at %s", db_path)
            return conn
            
        except sqlite3.Error as e:
            self.logger.critical("Database initialization failed: %s", e)
            raise
    
    def scan_directory(self, root_dir, recursive=True, exclusion_patterns=None, depth=0):
        """
        Recursively scan directories for video files.
        
        Args:
            root_dir (str): Root directory to scan
            recursive (bool): Whether to scan subdirectories
            exclusion_patterns (list): List of regex patterns for files/dirs to exclude
            depth (int): Current recursion depth (used internally)
            
        Returns:
            list: List of video file paths found
        """
        if exclusion_patterns is None:
            exclusion_patterns = self.config['exclusion_patterns']
            
        # Check recursion depth limit
        if self.config['max_recursion_depth'] is not None and depth > self.config['max_recursion_depth']:
            self.logger.info("Max recursion depth reached at %s", root_dir)
            return []
            
        start_time = time.time()
        self.logger.info("Scanning directory: %s", root_dir)
        
        video_files = []
        
        try:
            with os.scandir(root_dir) as entries:
                for entry in entries:
                    # Skip if matches exclusion pattern
                    if exclusion_patterns and any(re.search(pattern, entry.path) for pattern in exclusion_patterns):
                        self.logger.debug("Skipping excluded path: %s", entry.path)
                        continue
                        
                    if entry.is_file():
                        # Check if it's a video file by extension
                        file_ext = os.path.splitext(entry.name)[1].lower()
                        if file_ext in self.config['video_extensions']:
                            try:
                                # Get file size
                                file_size = entry.stat().st_size
                                
                                # Skip if file is too large
                                if file_size > self.config['max_file_size']:
                                    self.logger.warning("Skipping large file: %s (%d bytes)", 
                                                     entry.path, file_size)
                                    continue
                                
                                # Check if already processed
                                if not self.check_file_processed(entry.path, self.db_connection):
                                    video_files.append(entry.path)
                                    self.logger.debug("Found video: %s", entry.path)
                            except OSError as e:
                                self.logger.error("Error accessing file %s: %s", entry.path, e)
                    
                    elif entry.is_dir() and recursive:
                        # Recursively scan subdirectories
                        sub_videos = self.scan_directory(entry.path, recursive, exclusion_patterns, depth + 1)
                        video_files.extend(sub_videos)
        
        except PermissionError as e:
            self.logger.error("Permission denied accessing directory %s: %s", root_dir, e)
        except OSError as e:
            self.logger.error("OS error accessing directory %s: %s", root_dir, e)
            
        elapsed_time = time.time() - start_time
        self.logger.info("Found %d video files in %s (took %.2f seconds)", 
                      len(video_files), root_dir, elapsed_time)
        
        return video_files
    
    def get_file_metadata(self, file_path):
        """
        Extract creation date, modification date, size, and other metadata from a video file.
        This method attempts to extract basic video metadata without requiring external libraries.
        For more comprehensive metadata extraction, consider using libraries like OpenCV or FFmpeg.
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            dict: File metadata
        """
        try:
            file_stat = os.stat(file_path)
            file_path_obj = Path(file_path)
            
            # Calculate file hash for identifying duplicate files
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            file_hash = hash_md5.hexdigest()
            
            # Basic metadata
            metadata = {
                'file_path': file_path,
                'file_name': file_path_obj.name,
                'file_size': file_stat.st_size,
                'creation_date': datetime.datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modification_date': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'access_date': datetime.datetime.fromtimestamp(file_stat.st_atime).isoformat(),
                'file_hash': file_hash,
                'format': file_path_obj.suffix[1:],  # Remove the dot from extension
                'duration': None,
                'resolution': None
            }
            
            # Attempt to extract basic video information from file headers
            # This is a basic implementation. For more detailed extraction,
            # consider using a dedicated video processing library
            try:
                self._try_extract_video_info(file_path, metadata)
            except Exception as e:
                self.logger.debug("Could not extract detailed video metadata: %s", e)
            
            self.logger.debug("Extracted metadata for %s: %s", file_path, metadata)
            return metadata
            
        except OSError as e:
            self.logger.error("Error getting metadata for %s: %s", file_path, e)
            raise
    
    def _try_extract_video_info(self, file_path, metadata):
        """
        Attempt to extract basic video information from file headers.
        This is a very basic implementation that tries to identify some common formats.
        
        Args:
            file_path (str): Path to the video file
            metadata (dict): Metadata dictionary to update with video info
        """
        # This is a simplified approach. A robust implementation would use
        # a dedicated video processing library like OpenCV or FFmpeg
        with open(file_path, 'rb') as f:
            header = f.read(32)
            
            # Try to identify format and extract some basic info
            if header.startswith(b'\x00\x00\x00\x18ftypmp42'):
                # MP4 file - very basic detection
                metadata['format'] = 'mp4'
            elif header.startswith(b'\x00\x00\x00\x1cftyp'):
                # Possibly MOV or other MP4 variant
                metadata['format'] = 'mp4/mov'
            elif header.startswith(b'\x1aE\xdf\xa3'):
                # Matroska/WebM
                metadata['format'] = 'mkv/webm'
            elif header.startswith(b'RIFF') and b'AVI' in header:
                # AVI file
                metadata['format'] = 'avi'
            elif header.startswith(b'FLV'):
                # FLV file
                metadata['format'] = 'flv'
    
    def check_file_processed(self, file_path, db_connection):
        """
        Check if a file has already been processed.
        
        Args:
            file_path (str): Path to the video file
            db_connection (sqlite3.Connection): Database connection
            
        Returns:
            bool: True if file has been processed, False otherwise
        """
        try:
            cursor = db_connection.cursor()
            cursor.execute(
                "SELECT status FROM video_files WHERE file_path = ?", 
                (file_path,)
            )
            result = cursor.fetchone()
            
            if result is None:
                return False
                
            status = result[0]
            is_processed = status in ('success', 'failed')
            
            if is_processed:
                self.logger.debug("File %s already processed with status: %s", file_path, status)
            
            return is_processed
            
        except sqlite3.Error as e:
            self.logger.error("Database error checking processed status: %s", e)
            return False
    
    def update_file_status(self, file_path, status, processing_details, db_connection):
        """
        Update the processing status of a file in the database.
        
        Args:
            file_path (str): Path to the video file
            status (str): Processing status ('pending', 'processing', 'success', 'failed')
            processing_details (dict): Details about the processing
            db_connection (sqlite3.Connection): Database connection
        """
        try:
            cursor = db_connection.cursor()
            now = datetime.datetime.now().isoformat()
            
            # Check if file exists in the database
            cursor.execute("SELECT id FROM video_files WHERE file_path = ?", (file_path,))
            file_record = cursor.fetchone()
            
            if file_record is None:
                # File doesn't exist in the database, insert it
                metadata = self.get_file_metadata(file_path)
                
                cursor.execute('''
                    INSERT INTO video_files (
                        file_path, file_hash, file_size, creation_date, modification_date, 
                        format, duration, resolution, status, last_processed, error_message, processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_path, metadata['file_hash'], metadata['file_size'],
                    metadata['creation_date'], metadata['modification_date'],
                    metadata.get('format', None), metadata.get('duration', None), 
                    metadata.get('resolution', None), status, now,
                    processing_details.get('error_message', None),
                    processing_details.get('processing_time', None)
                ))
                
                file_id = cursor.lastrowid
            else:
                # File exists, update its status
                file_id = file_record[0]
                
                cursor.execute('''
                    UPDATE video_files 
                    SET status = ?, last_processed = ?, error_message = ?, processing_time = ?
                    WHERE id = ?
                ''', (
                    status, now,
                    processing_details.get('error_message', None),
                    processing_details.get('processing_time', None),
                    file_id
                ))
            
            # Insert into processing history to track all processing attempts
            cursor.execute('''
                INSERT INTO processing_history (
                    file_id, processed_date, status, processing_time, error_message
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                file_id, now, status,
                processing_details.get('processing_time', None),
                processing_details.get('error_message', None)
            ))
            
            db_connection.commit()
            self.logger.debug("Updated status for %s: %s", file_path, status)
            
        except sqlite3.Error as e:
            self.logger.error("Database error updating file status: %s", e)
            db_connection.rollback()
        except Exception as e:
            self.logger.error("Error updating file status: %s", e)
            db_connection.rollback()
    
    def create_processing_queue(self, file_list, batch_size=None):
        """
        Create a processing queue with specified batch size.
        
        Args:
            file_list (list): List of video file paths
            batch_size (int, optional): Batch size. If None, use config value.
            
        Returns:
            int: Number of files added to the queue
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
            
        count = 0
        self.logger.info("Creating processing queue with %d files", len(file_list))
        
        try:
            # Clear existing queue
            while not self.processing_queue.empty():
                self.processing_queue.get()
            
            # Get file modification times for prioritization
            file_priorities = []
            for file_path in file_list:
                try:
                    mod_time = os.path.getmtime(file_path)
                    # Priority is negative modification time (newer files processed first)
                    priority = -mod_time
                    file_priorities.append((priority, file_path))
                except OSError as e:
                    self.logger.error("Error accessing file %s: %s", file_path, e)
            
            # Sort by priority (newer files first)
            file_priorities.sort()
            
            # Add to queue, up to batch_size
            for i, (priority, file_path) in enumerate(file_priorities):
                if batch_size and i >= batch_size:
                    break
                self.processing_queue.put((priority, file_path))
                count += 1
                
            self.processing_status['total_files'] = count
            self.logger.info("Added %d files to processing queue", count)
            
            return count
            
        except Exception as e:
            self.logger.error("Error creating processing queue: %s", e)
            return 0
    
    def validate_video(self, file_path):
        """
        Check if a file is a valid video file and can be opened.
        This method does basic validation by checking file extension and header.
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.config['video_extensions']:
                return False, f"Not a recognized video format: {file_ext}"
            
            # Try to open the file and read the first few bytes
            with open(file_path, 'rb') as f:
                header = f.read(32)
            
            # Check for common video file signatures
            # This is a very basic check - a real implementation would use a dedicated video library
            if (header.startswith(b'\x00\x00\x00\x18ftypmp42') or  # MP4
                header.startswith(b'\x00\x00\x00\x1cftyp') or      # MP4/MOV
                header.startswith(b'\x1aE\xdf\xa3') or             # MKV/WebM
                header.startswith(b'RIFF') and b'AVI' in header or # AVI
                header.startswith(b'FLV')):                        # FLV
                return True, None
            
            # For other formats, just check if the file is not empty
            if os.path.getsize(file_path) > 1000:  # Arbitrary minimum size
                return True, None
            
            return False, "File does not appear to be a valid video (invalid header)"
                
        except Exception as e:
            return False, str(e)
    
    def handle_processing_error(self, file_path, error, db_connection):
        """
        Log error details and update database when processing fails.
        
        Args:
            file_path (str): Path to the video file
            error (Exception): The exception that occurred
            db_connection (sqlite3.Connection): Database connection
        """
        error_message = f"{type(error).__name__}: {str(error)}"
        stacktrace = traceback.format_exc()
        
        self.logger.error("Error processing %s: %s", file_path, error_message)
        self.logger.debug("Stacktrace: %s", stacktrace)
        
        processing_details = {
            'error_message': error_message,
            'processing_time': None
        }
        
        self.update_file_status(file_path, 'failed', processing_details, db_connection)
        self.processing_status['failure_count'] += 1
    
    def generate_processing_report(self, start_time, end_time, success_count, failure_count):
        """
        Generate a summary report of the processing job.
        
        Args:
            start_time (datetime): Job start time
            end_time (datetime): Job end time
            success_count (int): Number of successfully processed files
            failure_count (int): Number of failed files
            
        Returns:
            dict: Report data
        """
        duration = (end_time - start_time).total_seconds()
        total_files = success_count + failure_count
        
        report = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_files': total_files,
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': (success_count / total_files * 100) if total_files > 0 else 0
        }
        
        # Save report to database
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO processing_jobs (
                    start_time, end_time, total_files, successful_files, failed_files, job_status
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                report['start_time'], report['end_time'], 
                report['total_files'], report['success_count'], report['failure_count'],
                'completed'
            ))
            self.db_connection.commit()
        except sqlite3.Error as e:
            self.logger.error("Error saving job report to database: %s", e)
        
        # Create a human-readable report string
        report_str = "\n".join([
            "===== Processing Job Report =====",
            f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {duration:.2f} seconds",
            f"Total files: {total_files}",
            f"Successful: {success_count} ({report['success_rate']:.1f}%)",
            f"Failed: {failure_count}",
            "=================================="
        ])
        
        self.logger.info("Processing job completed:\n%s", report_str)
        return report
    
    def cleanup_temp_files(self, temp_dir=None):
        """
        Clean up any temporary files created during processing.
        
        Args:
            temp_dir (str, optional): Directory containing temp files. If None, use config.
        """
        if temp_dir is None:
            temp_dir = self.config['temp_dir']
            
        try:
            if os.path.exists(temp_dir):
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        self.logger.error("Error removing temp item %s: %s", item_path, e)
                        
                self.logger.info("Temporary files cleaned up from %s", temp_dir)
            else:
                self.logger.warning("Temp directory %s does not exist", temp_dir)
                
        except Exception as e:
            self.logger.error("Error cleaning up temp files: %s", e)
    
    def process_queue_worker(self, process_func):
        """
        Worker function to process files from the queue.
        
        Args:
            process_func (callable): Function to process a single video file
        """
        while not self.processing_queue.empty() and self.processing_status['running']:
            try:
                # Get next file from queue
                _, file_path = self.processing_queue.get(block=False)
                
                self.processing_status['current_file'] = file_path
                self.logger.info("Processing file: %s", file_path)
                
                # Update status in database
                self.update_file_status(file_path, 'processing', {}, self.db_connection)
                
                # Process the file
                start_time = time.time()
                try:
                    # Validate video before processing
                    is_valid, error_message = self.validate_video(file_path)
                    if not is_valid:
                        raise ValueError(f"Invalid video file: {error_message}")
                    
                    # Process the video using the provided function
                    result = process_func(file_path)
                    
                    # Update status on success
                    processing_time = time.time() - start_time
                    processing_details = {
                        'processing_time': processing_time,
                        'error_message': None
                    }
                    
                    self.update_file_status(file_path, 'success', processing_details, self.db_connection)
                    self.processing_status['success_count'] += 1
                    
                    self.logger.info("Successfully processed %s in %.2f seconds", 
                                   file_path, processing_time)
                    
                except Exception as e:
                    # Handle processing error
                    self.handle_processing_error(file_path, e, self.db_connection)
                
                # Update processed count
                self.processing_status['processed_files'] += 1
                
            except queue.Empty:
                # Queue is empty
                break
                
            except Exception as e:
                self.logger.error("Error in processing worker: %s", e)
    
    def process_files(self, file_list=None, process_func=None, num_threads=None):
        """
        Process a list of video files using the provided function.
        
        Args:
            file_list (list, optional): List of video file paths. If None, scan directory.
            process_func (callable, optional): Function to process a single video file.
                Should take a file path as its argument.
            num_threads (int, optional): Number of processing threads. If None, use config.
                
        Returns:
            dict: Processing report
        """
        if num_threads is None:
            num_threads = self.config['threads']
            
        if process_func is None:
            # Define a dummy processing function if none provided
            def process_func(file_path):
                self.logger.info("Dummy processing of %s", file_path)
                return True
        
        # Reset status tracking
        self.processing_status = {
            'running': True,
            'total_files': 0,
            'processed_files': 0,
            'success_count': 0,
            'failure_count': 0,
            'current_file': None,
            'start_time': datetime.datetime.now(),
            'end_time': None
        }
        
        try:
            # Create processing queue
            if file_list is None:
                # Scan directory if no file list provided
                file_list = self.scan_directory(self.config.get('scan_dir', '.'), 
                                             recursive=True)
            
            files_queued = self.create_processing_queue(file_list)
            
            if files_queued == 0:
                self.logger.warning("No files to process")
                self.processing_status['end_time'] = datetime.datetime.now()
                return self.generate_processing_report(
                    self.processing_status['start_time'],
                    self.processing_status['end_time'],
                    0, 0
                )
            
            # Start worker threads for parallel processing
            threads = []
            for _ in range(min(num_threads, files_queued)):
                thread = threading.Thread(target=self.process_queue_worker, args=(process_func,))
                thread.daemon = True
                thread.start()
                threads.append(thread)
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
                
            # Update status
            self.processing_status['running'] = False
            self.processing_status['end_time'] = datetime.datetime.now()
            
            # Generate and return report
            return self.generate_processing_report(
                self.processing_status['start_time'],
                self.processing_status['end_time'],
                self.processing_status['success_count'],
                self.processing_status['failure_count']
            )
            
        except Exception as e:
            self.logger.error("Error in process_files: %s", e)
            self.processing_status['running'] = False
            self.processing_status['end_time'] = datetime.datetime.now()
            
            return {
                'error': str(e),
                'start_time': self.processing_status['start_time'].isoformat(),
                'end_time': self.processing_status['end_time'].isoformat()
            }
    
    def get_processing_status(self):
        """
        Get the current processing status.
        
        Returns:
            dict: Status information including progress percentage and time estimates
        """
        status = self.processing_status.copy()
        
        # Calculate progress percentage
        if status['total_files'] > 0:
            status['progress_percent'] = (status['processed_files'] / status['total_files']) * 100
        else:
            status['progress_percent'] = 0
            
        # Calculate elapsed time
        if status['start_time']:
            if status['end_time']:
                end_time = status['end_time']
            else:
                end_time = datetime.datetime.now()
                
            status['elapsed_seconds'] = (end_time - status['start_time']).total_seconds()
            
            # Estimate remaining time
            if status['progress_percent'] > 0:
                status['estimated_total_seconds'] = status['elapsed_seconds'] / (status['progress_percent'] / 100)
                status['estimated_remaining_seconds'] = status['estimated_total_seconds'] - status['elapsed_seconds']
            else:
                status['estimated_remaining_seconds'] = None
        
        return status
    
    def pause_processing(self):
        """
        Pause the processing queue.
        
        Returns:
            bool: True if paused successfully
        """
        if self.processing_status['running']:
            self.processing_status['running'] = False
            self.logger.info("Processing paused")
            return True
        else:
            self.logger.info("Processing already paused")
            return False
    
    def resume_processing(self, process_func, num_threads=None):
        """
        Resume the processing queue.
        
        Args:
            process_func (callable): Function to process a single video file
            num_threads (int, optional): Number of processing threads. If None, use config.
            
        Returns:
            bool: True if resumed successfully
        """
        if not self.processing_status['running']:
            self.processing_status['running'] = True
            
            if num_threads is None:
                num_threads = self.config['threads']
                
            # Start worker threads
            remaining_files = self.processing_status['total_files'] - self.processing_status['processed_files']
            threads = []
            for _ in range(min(num_threads, remaining_files)):
                thread = threading.Thread(target=self.process_queue_worker, args=(process_func,))
                thread.daemon = True
                thread.start()
                threads.append(thread)
                
            self.logger.info("Processing resumed with %d threads", len(threads))
            return True
        else:
            self.logger.info("Processing already running")
            return False
    
    def get_supported_formats(self):
        """
        Get a list of supported video file formats.
        
        Returns:
            list: List of supported video file extensions
        """
        return self.config['video_extensions']
    
    def add_supported_format(self, extension):
        """
        Add a new video file format to the supported formats list.
        
        Args:
            extension (str): File extension (e.g., '.mp4')
            
        Returns:
            bool: True if added successfully, False if already exists
        """
        if not extension.startswith('.'):
            extension = '.' + extension
            
        extension = extension.lower()
        
        if extension in self.config['video_extensions']:
            self.logger.debug("Format %s already supported", extension)
            return False
            
        self.config['video_extensions'].append(extension)
        self.logger.info("Added new supported format: %s", extension)
        return True
    
    def close(self):
        """
        Close the Video processor, releasing resources.
        """
        try:
            if hasattr(self, 'db_connection') and self.db_connection:
                self.db_connection.close()
                
            self.logger.info("Video Processor closed")
            
            # Close log handlers
            if hasattr(self, 'logger') and self.logger:
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
                    
        except Exception as e:
            print(f"Error closing Video Processor: {e}")


# Example usage of the Video Processor
def example_usage():
    """Example demonstrating the use of VideoProcessor class"""
    # Define a custom video processing function
    def process_video(file_path):
        """
        Example function to process a video file.
        This would be where you implement your actual video processing logic.
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            bool: True if processing was successful
        """
        print(f"Processing video: {file_path}")
        # Simulate processing time
        time.sleep(1)
        return True
    
    # Configuration for the video processor
    config = {
        'log_dir': './logs',
        'log_level': logging.INFO,
        'db_path': './video_tracking.db',
        'scan_dir': './video_samples',  # Directory with videos to scan
        'batch_size': 5,
        'threads': 2
    }
    
    # Initialize the processor
    processor = VideoProcessor(config)
    
    try:
        # Scan for video files
        video_files = processor.scan_directory(config['scan_dir'])
        
        if video_files:
            print(f"Found {len(video_files)} video files")
            
            # Process the files
            report = processor.process_files(video_files, process_video)
            
            # Print the report
            print(f"Processing completed. Success: {report['success_count']}, Failed: {report['failure_count']}")
            
            # Get status
            status = processor.get_processing_status()
            print(f"Progress: {status['progress_percent']:.1f}%")
        else:
            print("No video files found")
            
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Close the processor
        processor.close()

if __name__ == "__main__":
    # Run the example if this file is executed directly
    example_usage()