#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the AudioProcessor class.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile

from src.video2docs import AudioProcessor


class TestAudioProcessor(unittest.TestCase):
    """Test cases for the AudioProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.audio_processor = AudioProcessor(use_gpu=False)
        self.temp_dir = tempfile.mkdtemp()
        self.audio_path = os.path.join(self.temp_dir, "test_audio.wav")

    @patch("src.video2docs.AudioSegment.from_file")
    @patch("src.video2docs.sr.AudioFile")
    def test_transcribe_audio(self, mock_audio_file, mock_from_file):
        """Test transcribing audio to text."""
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 120000  # 120 seconds
        mock_audio.__getitem__.return_value = mock_audio
        mock_audio.export.return_value = MagicMock()
        mock_from_file.return_value = mock_audio
        
        # Mock AudioFile context manager
        mock_source = MagicMock()
        mock_audio_file.return_value.__enter__.return_value = mock_source
        
        # Mock Recognizer
        self.audio_processor.recognizer = MagicMock()
        self.audio_processor.recognizer.record.return_value = "audio_data"
        self.audio_processor.recognizer.recognize_google.return_value = "This is a test transcription."
        
        # Mock os.remove to avoid actual file deletion
        with patch("src.video2docs.os.remove") as mock_remove:
            # Call the method
            result = self.audio_processor.transcribe_audio(self.audio_path, chunk_size=60000)
            
            # Assertions
            self.assertEqual(len(result), 2)  # Two chunks (120000 / 60000 = 2)
            
            # Check first chunk
            self.assertEqual(result[0]["text"], "This is a test transcription.")
            self.assertEqual(result[0]["start_time"], 0.0)
            self.assertEqual(result[0]["end_time"], 60.0)
            
            # Check second chunk
            self.assertEqual(result[1]["text"], "This is a test transcription.")
            self.assertEqual(result[1]["start_time"], 60.0)
            self.assertEqual(result[1]["end_time"], 120.0)
            
            # Check that AudioSegment.from_file was called
            mock_from_file.assert_called_once_with(self.audio_path)
            
            # Check that audio was exported for each chunk
            self.assertEqual(mock_audio.export.call_count, 2)
            
            # Check that AudioFile was created for each chunk
            self.assertEqual(mock_audio_file.call_count, 2)
            
            # Check that recognize_google was called for each chunk
            self.assertEqual(self.audio_processor.recognizer.recognize_google.call_count, 2)
            
            # Check that temporary files were removed
            self.assertEqual(mock_remove.call_count, 2)

    @patch("src.video2docs.AudioSegment.from_file")
    def test_transcribe_audio_empty(self, mock_from_file):
        """Test transcribing empty audio."""
        # Mock AudioSegment with zero length
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 0  # 0 seconds
        mock_from_file.return_value = mock_audio
        
        # Call the method
        result = self.audio_processor.transcribe_audio(self.audio_path)
        
        # Assertions
        self.assertEqual(len(result), 0)  # No chunks
        
        # Check that AudioSegment.from_file was called
        mock_from_file.assert_called_once_with(self.audio_path)

    @patch("src.video2docs.AudioSegment.from_file")
    @patch("src.video2docs.sr.AudioFile")
    def test_transcribe_audio_exception(self, mock_audio_file, mock_from_file):
        """Test handling exceptions during transcription."""
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 60000  # 60 seconds
        mock_audio.__getitem__.return_value = mock_audio
        mock_audio.export.return_value = MagicMock()
        mock_from_file.return_value = mock_audio
        
        # Mock AudioFile context manager
        mock_source = MagicMock()
        mock_audio_file.return_value.__enter__.return_value = mock_source
        
        # Mock Recognizer to raise an exception
        self.audio_processor.recognizer = MagicMock()
        self.audio_processor.recognizer.record.return_value = "audio_data"
        self.audio_processor.recognizer.recognize_google.side_effect = Exception("Recognition error")
        
        # Mock os.remove to avoid actual file deletion
        with patch("src.video2docs.os.remove") as mock_remove:
            # Check that the method raises the exception
            with self.assertRaises(Exception):
                self.audio_processor.transcribe_audio(self.audio_path)
            
            # Check that temporary files were removed
            self.assertEqual(mock_remove.call_count, 1)


if __name__ == "__main__":
    unittest.main()