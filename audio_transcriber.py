import os
import logging
import datetime
import time
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Try to import whisper, providing a helpful error if not installed
try:
    import whisper
except ImportError:
    raise ImportError(
        "OpenAI Whisper is required for this library. "
        "Please install it with: pip install openai-whisper"
    )

class AudioTranscriber:
    """
    Audio Transcriber library for converting speech in audio files to text.
    This module focuses solely on transcribing audio files using OpenAI's Whisper
    and saving transcripts with timestamps in text format.
    """
    def __init__(self, config=None):
        """
        Initialize the Audio Transcriber with the provided configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with transcription parameters.
                If None, default configuration will be used.
        """
        # Default configuration optimized for balanced accuracy and speed
        self.config = {
            'output_dir': './transcripts',
            'model_size': 'base',     # Options: 'tiny', 'base', 'small', 'medium', 'large'
            'language': None,         # Auto-detect language if None, or specify (e.g., 'en', 'fr')
            'timestamp_format': '%H:%M:%S',  # Format of timestamps in output
            'ffmpeg_path': 'ffmpeg',  # Path to ffmpeg executable
            'device': 'cpu',          # 'cpu' or 'cuda' (for GPU acceleration)
            'log_level': logging.INFO,
            'beam_size': 5,           # Beam size for decoding (larger = more accurate but slower)
            'compute_type': 'float16', # Type for computation (float16 for GPU, float32 for CPU)
            'verbose': False,         # Whether to print detailed progress
            'return_timestamps': True # Whether to include timestamps in output
        }
        
        # Update with user-provided config
        if config:
            self.config.update(config)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Ensure output directory exists
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Load the Whisper model
        self.model = None
        self._load_model()
        
        self.logger.info(f"Audio Transcriber initialized with model size: {self.config['model_size']}")
    
    def _setup_logging(self):
        """
        Set up basic logging configuration.
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('audio_transcriber')
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
                file_handler = logging.FileHandler(os.path.join(log_dir, 'transcription.log'))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not set up file logging: {e}")
        
        return logger
    
    def _load_model(self):
        """
        Load the Whisper model based on configuration.
        """
        try:
            self.logger.info(f"Loading Whisper {self.config['model_size']} model...")
            start_time = time.time()
            
            # Try different parameter combinations based on what's supported
            try:
                # First try with all parameters (newer versions)
                self.model = whisper.load_model(
                    self.config['model_size'],
                    device=self.config['device'],
                    compute_type=self.config['compute_type']
                )
            except TypeError:
                try:
                    # Try without compute_type (mid versions)
                    self.model = whisper.load_model(
                        self.config['model_size'],
                        device=self.config['device']
                    )
                except TypeError:
                    # Fallback to just the model name (oldest versions)
                    self.model = whisper.load_model(
                        self.config['model_size']
                    )
                    # Manually set device after loading if possible
                    if hasattr(self.model, "to") and self.config['device'] != 'cpu':
                        self.model = self.model.to(self.config['device'])
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: str, output_path: Optional[str] = None, 
                         language: Optional[str] = None) -> Dict:
        """
        Transcribe an audio file using the Whisper model.
        
        Args:
            audio_path (str): Path to the audio file to transcribe
            output_path (str, optional): Path where to save the transcript.
                If None, will be generated based on audio path and output directory.
            language (str, optional): Language code to use for transcription.
                If None, use the value from config (which might be auto-detection).
                
        Returns:
            dict: Transcription result containing text, segments with timestamps, etc.
        """
        if not os.path.exists(audio_path):
            error_msg = f"Audio file does not exist: {audio_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Generate output path if not provided
        if output_path is None:
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(
                self.config['output_dir'],
                f"{audio_name}_transcript.txt"
            )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use language from parameters or config
        lang = language or self.config['language']
        
        try:
            self.logger.info(f"Transcribing {audio_path}...")
            start_time = time.time()
            
            # Set the transcription options
            transcribe_options = {
                "language": lang,
                "beam_size": self.config['beam_size'],
                "verbose": self.config['verbose']
            }
            
            # Add timestamp options if needed
            if self.config['return_timestamps']:
                transcribe_options["task"] = "transcribe"
                
            # Perform the transcription
            result = self.model.transcribe(audio_path, **transcribe_options)
            
            transcription_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            
            # Save the transcript to a text file
            self._save_transcript(result, output_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            raise
    
    def _save_transcript(self, transcription_result: Dict, output_path: str) -> str:
        """
        Save transcription result to a text file.
        
        Args:
            transcription_result (dict): The transcription result from Whisper
            output_path (str): Path where to save the transcript
            
        Returns:
            str: Path to the saved transcript file
        """
        try:
            self.logger.info(f"Saving transcript to {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write a header
                f.write(f"Transcript created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Language: {transcription_result.get('language', 'auto-detected')}\n")
                f.write("-" * 80 + "\n\n")
                
                # Write each segment with timestamp
                for segment in transcription_result.get('segments', []):
                    # Format the start time
                    start_time = segment.get('start', 0)
                    formatted_time = self._format_timestamp(start_time)
                    
                    # Write the segment with its timestamp
                    f.write(f"[{formatted_time}] {segment.get('text', '').strip()}\n")
                    
                    # Optional: Add a newline between paragraphs or after certain punctuation
                    if segment.get('text', '').strip().endswith(('.', '!', '?')):
                        f.write("\n")
            
            self.logger.info(f"Transcript saved successfully")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving transcript: {e}")
            raise
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format time in seconds to a timestamp string.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted timestamp
        """
        timestamp_format = self.config['timestamp_format']
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if '%H' in timestamp_format:
            return datetime.time(int(hours), int(minutes), int(seconds)).strftime(timestamp_format)
        else:
            # For formats without hours
            return datetime.time(0, int(minutes), int(seconds)).strftime(timestamp_format)
    
    def batch_transcribe(self, audio_files: List[str], language: Optional[str] = None) -> List[Dict]:
        """
        Transcribe a batch of audio files.
        
        Args:
            audio_files (list): List of paths to audio files
            language (str, optional): Language code to use for all files
            
        Returns:
            list: List of transcription results
        """
        results = []
        
        for audio_path in audio_files:
            try:
                result = self.transcribe_audio(audio_path, language=language)
                results.append({
                    'audio_path': audio_path,
                    'output_path': os.path.join(
                        self.config['output_dir'],
                        f"{os.path.splitext(os.path.basename(audio_path))[0]}_transcript.txt"
                    ),
                    'success': True,
                    'language': result.get('language')
                })
            except Exception as e:
                self.logger.error(f"Failed to transcribe {audio_path}: {e}")
                results.append({
                    'audio_path': audio_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def transcribe_with_diarization(self, audio_path: str, output_path: Optional[str] = None,
                                   num_speakers: Optional[int] = None) -> Dict:
        """
        Transcribe with speaker diarization (who said what).
        Note: This requires additional dependencies (pyannote.audio).
        
        Args:
            audio_path (str): Path to the audio file
            output_path (str, optional): Path to save the transcript
            num_speakers (int, optional): Number of speakers to detect (if known)
            
        Returns:
            dict: Transcription result with speaker information
        """
        try:
            # Check if pyannote.audio is installed
            import pyannote.audio
        except ImportError:
            raise ImportError(
                "Speaker diarization requires pyannote.audio. "
                "Please install it with: pip install pyannote.audio"
            )
        
        self.logger.info(f"Starting transcription with speaker diarization for {audio_path}")
        
        # First get the regular transcription with timestamps
        result = self.transcribe_audio(audio_path, output_path=None, language=self.config['language'])
        
        try:
            # Initialize the diarization pipeline
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
            
            # Perform diarization
            diarization = pipeline(audio_path, num_speakers=num_speakers)
            
            # Assign speakers to segments
            speaker_segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'speaker': speaker
                })
            
            # Combine with transcription by matching timestamps
            for trans_segment in result['segments']:
                # Find overlapping speaker segments
                segment_start = trans_segment['start']
                segment_end = segment_start + trans_segment['end'] - trans_segment['start']
                
                matching_speakers = []
                for s in speaker_segments:
                    # Check for overlap
                    if max(segment_start, s['start']) < min(segment_end, s['end']):
                        matching_speakers.append(s['speaker'])
                
                # Add the most likely speaker
                if matching_speakers:
                    # Use the most common speaker in this segment
                    from collections import Counter
                    speaker_counts = Counter(matching_speakers)
                    trans_segment['speaker'] = speaker_counts.most_common(1)[0][0]
                else:
                    trans_segment['speaker'] = "Unknown"
            
            # Save with speaker information if output path is provided
            if output_path:
                self._save_transcript_with_speakers(result, output_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Diarization error: {e}")
            # Fall back to regular transcription
            self.logger.info("Falling back to regular transcription without speaker identification")
            return result
    
    def _save_transcript_with_speakers(self, transcription_result: Dict, output_path: str) -> str:
        """
        Save transcription result with speaker information to a text file.
        
        Args:
            transcription_result (dict): The transcription result with speaker info
            output_path (str): Path where to save the transcript
            
        Returns:
            str: Path to the saved transcript file
        """
        try:
            self.logger.info(f"Saving transcript with speakers to {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write a header
                f.write(f"Transcript with speaker diarization: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Language: {transcription_result.get('language', 'auto-detected')}\n")
                f.write("-" * 80 + "\n\n")
                
                # Write each segment with timestamp and speaker
                for segment in transcription_result.get('segments', []):
                    # Format the start time
                    start_time = segment.get('start', 0)
                    formatted_time = self._format_timestamp(start_time)
                    
                    # Get speaker info (if available)
                    speaker = segment.get('speaker', 'Unknown')
                    
                    # Write the segment with its timestamp and speaker
                    f.write(f"[{formatted_time}] {speaker}: {segment.get('text', '').strip()}\n")
                    
                    # Optional: Add a newline between paragraphs or after certain punctuation
                    if segment.get('text', '').strip().endswith(('.', '!', '?')):
                        f.write("\n")
            
            self.logger.info(f"Transcript with speakers saved successfully")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving transcript with speakers: {e}")
            raise
    
    def export_transcript_formats(self, transcription_result: Dict, output_base_path: str) -> Dict:
        """
        Export the transcript to multiple formats (txt, json, srt, vtt).
        
        Args:
            transcription_result (dict): The transcription result
            output_base_path (str): Base path for the output files (without extension)
            
        Returns:
            dict: Dictionary of paths to the exported files
        """
        results = {}
        
        # Base paths
        txt_path = f"{output_base_path}.txt"
        json_path = f"{output_base_path}.json"
        srt_path = f"{output_base_path}.srt"
        vtt_path = f"{output_base_path}.vtt"
        
        # Save TXT (already implemented)
        self._save_transcript(transcription_result, txt_path)
        results['txt'] = txt_path
        
        # Save JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_result, f, indent=2, ensure_ascii=False)
            results['json'] = json_path
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")
        
        # Save SRT
        try:
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(transcription_result.get('segments', []), 1):
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', start_time)
                    
                    # Format timestamps for SRT (00:00:00,000)
                    start_formatted = self._format_srt_timestamp(start_time)
                    end_formatted = self._format_srt_timestamp(end_time)
                    
                    # Write the SRT entry
                    f.write(f"{i}\n")
                    f.write(f"{start_formatted} --> {end_formatted}\n")
                    f.write(f"{segment.get('text', '').strip()}\n\n")
            
            results['srt'] = srt_path
        except Exception as e:
            self.logger.error(f"Error saving SRT: {e}")
        
        # Save VTT
        try:
            with open(vtt_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for segment in transcription_result.get('segments', []):
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', start_time)
                    
                    # Format timestamps for VTT (00:00:00.000)
                    start_formatted = self._format_vtt_timestamp(start_time)
                    end_formatted = self._format_vtt_timestamp(end_time)
                    
                    # Write the VTT entry
                    f.write(f"{start_formatted} --> {end_formatted}\n")
                    f.write(f"{segment.get('text', '').strip()}\n\n")
            
            results['vtt'] = vtt_path
        except Exception as e:
            self.logger.error(f"Error saving VTT: {e}")
        
        return results
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        Format time in seconds to SRT timestamp format (00:00:00,000).
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted SRT timestamp
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def _format_vtt_timestamp(self, seconds: float) -> str:
        """
        Format time in seconds to VTT timestamp format (00:00:00.000).
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted VTT timestamp
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"


# Example usage
def example_usage():
    """Demonstrate how to use the AudioTranscriber class"""
    # Initialize the transcriber with default settings
    transcriber = AudioTranscriber({
        'model_size': 'base',  # Use 'tiny', 'base', 'small', 'medium', or 'large'
        'output_dir': './transcripts',
        'device': 'cpu'  # Use 'cuda' for GPU acceleration if available
    })
    
    # Transcribe a single audio file
    result = transcriber.transcribe_audio('sample_audio.mp3')
    
    # Print some information about the transcription
    print(f"Detected language: {result['language']}")
    print(f"Transcript saved to: {os.path.join(transcriber.config['output_dir'], 'sample_audio_transcript.txt')}")
    
    # Batch transcribe multiple files
    batch_results = transcriber.batch_transcribe([
        'interview1.mp3',
        'lecture2.wav',
        'meeting3.m4a'
    ])
    
    # Export to multiple formats (TXT, JSON, SRT, VTT)
    export_paths = transcriber.export_transcript_formats(
        result,
        os.path.join(transcriber.config['output_dir'], 'sample_audio')
    )
    
    print(f"Exported formats: {export_paths}")

if __name__ == "__main__":
    # Run the example if this file is executed directly
    example_usage()