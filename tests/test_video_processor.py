#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the VideoProcessor class.
"""

import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil

from src.video2docs import VideoProcessor


class TestVideoProcessor(unittest.TestCase):
    """Test cases for the VideoProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.video_processor = VideoProcessor(temp_dir=self.temp_dir)
        
        # Create a sample frame for testing
        self.sample_frame_path = os.path.join(self.temp_dir, "sample_frame.jpg")
        with open(self.sample_frame_path, "w") as f:
            f.write("mock image data")

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("src.video2docs.pytube.YouTube")
    def test_download_youtube_video(self, mock_youtube):
        """Test downloading a YouTube video."""
        # Mock the YouTube object and its methods
        mock_stream = MagicMock()
        mock_stream.download.return_value = os.path.join(self.temp_dir, "test_video.mp4")
        
        mock_streams = MagicMock()
        mock_streams.filter.return_value = mock_streams
        mock_streams.order_by.return_value = mock_streams
        mock_streams.desc.return_value = mock_streams
        mock_streams.first.return_value = mock_stream
        
        mock_youtube.return_value.streams = mock_streams
        
        # Call the method
        result = self.video_processor.download_youtube_video("https://www.youtube.com/watch?v=test")
        
        # Assertions
        self.assertEqual(result, os.path.join(self.temp_dir, "test_video.mp4"))
        mock_youtube.assert_called_once_with("https://www.youtube.com/watch?v=test")
        mock_streams.filter.assert_called_once_with(progressive=True, file_extension='mp4')
        mock_stream.download.assert_called_once_with(output_path=self.temp_dir)

    @patch("src.video2docs.VideoFileClip")
    def test_extract_audio(self, mock_video_file_clip):
        """Test extracting audio from a video file."""
        # Mock the VideoFileClip object and its methods
        mock_video = MagicMock()
        mock_video.audio = MagicMock()
        mock_video_file_clip.return_value = mock_video
        
        # Call the method
        result = self.video_processor.extract_audio("test_video.mp4")
        
        # Assertions
        self.assertEqual(result, os.path.join(self.temp_dir, "audio.wav"))
        mock_video_file_clip.assert_called_once_with("test_video.mp4")
        mock_video.audio.write_audiofile.assert_called_once_with(
            os.path.join(self.temp_dir, "audio.wav"), codec='pcm_s16le'
        )

    @patch("src.video2docs.cv2.VideoCapture")
    @patch("src.video2docs.cv2.imwrite")
    def test_extract_frames(self, mock_imwrite, mock_video_capture):
        """Test extracting frames from a video file."""
        # Mock the VideoCapture object and its methods
        mock_video = MagicMock()
        mock_video.get.return_value = 30.0  # 30 fps
        
        # Set up the read method to return True, frame for first call and False, None for second call
        mock_video.read.side_effect = [(True, "frame1"), (False, None)]
        
        mock_video_capture.return_value = mock_video
        
        # Call the method
        result = self.video_processor.extract_frames("test_video.mp4", interval=1.0)
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 0.0)  # timestamp
        self.assertTrue(result[0][1].startswith(self.temp_dir))  # frame path
        
        mock_video_capture.assert_called_once_with("test_video.mp4")
        mock_video.get.assert_called_once_with(mock_video_capture.CAP_PROP_FPS)
        mock_imwrite.assert_called_once()
        mock_video.release.assert_called_once()

    @patch("src.video2docs.cv2.imread")
    @patch("src.video2docs.cv2.resize")
    @patch("src.video2docs.ssim")
    def test_detect_slides(self, mock_ssim, mock_resize, mock_imread):
        """Test detecting slides from frames."""
        # Mock the imread, resize, and ssim functions
        mock_imread.return_value = "image_data"
        mock_resize.return_value = "resized_image_data"
        
        # Set up ssim to return values that will trigger slide detection
        mock_ssim.side_effect = [0.7, 0.9]  # First comparison below threshold, second above
        
        # Create test frames
        frames = [
            (0.0, os.path.join(self.temp_dir, "frame_0000.jpg")),
            (1.0, os.path.join(self.temp_dir, "frame_0001.jpg")),
            (2.0, os.path.join(self.temp_dir, "frame_0002.jpg"))
        ]
        
        # Call the method
        result = self.video_processor.detect_slides(frames, threshold=0.8)
        
        # Assertions
        self.assertEqual(len(result), 2)  # First frame is always included, plus one more below threshold
        self.assertEqual(result[0], frames[0])
        self.assertEqual(result[1], frames[1])  # Second frame is below threshold (0.7 < 0.8)
        
        # Check that imread was called for each comparison (2 times)
        self.assertEqual(mock_imread.call_count, 4)
        
        # Check that ssim was called for each comparison (2 times)
        self.assertEqual(mock_ssim.call_count, 2)

    def test_detect_slides_empty_frames(self):
        """Test detecting slides with empty frames list."""
        result = self.video_processor.detect_slides([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()