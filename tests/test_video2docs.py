#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration tests for the Video2Docs class.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import os
import tempfile
import shutil

from src.video2docs import Video2Docs


class TestVideo2Docs(unittest.TestCase):
    """Integration test cases for the Video2Docs class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a sample video file
        self.video_path = os.path.join(self.temp_dir, "test_video.mp4")
        with open(self.video_path, "w") as f:
            f.write("mock video data")

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)

    @patch("src.video2docs.VideoProcessor")
    @patch("src.video2docs.AudioProcessor")
    @patch("src.video2docs.LLMProcessor")
    @patch("src.video2docs.DocumentGenerator")
    def test_process_local_video(self, mock_doc_gen, mock_llm, mock_audio, mock_video):
        """Test processing a local video file."""
        # Mock VideoProcessor
        mock_video_instance = MagicMock()
        mock_video.return_value = mock_video_instance
        mock_video_instance.extract_audio.return_value = "audio.wav"
        mock_video_instance.extract_frames.return_value = [
            (0.0, "frame1.jpg"),
            (1.0, "frame2.jpg")
        ]
        mock_video_instance.detect_slides.return_value = [
            (0.0, "frame1.jpg")
        ]
        
        # Mock AudioProcessor
        mock_audio_instance = MagicMock()
        mock_audio.return_value = mock_audio_instance
        mock_audio_instance.transcribe_audio.return_value = [
            {"text": "This is a test transcription.", "start_time": 0.0, "end_time": 30.0}
        ]
        
        # Mock LLMProcessor
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.organize_content.return_value = {
            "title": "Test Document",
            "summary": "This is a test summary.",
            "sections": [
                {
                    "heading": "Section 1",
                    "content": "This is section 1 content. [SLIDE 0]",
                    "bullet_points": ["Point 1", "Point 2"]
                }
            ]
        }
        
        # Mock DocumentGenerator
        mock_doc_gen_instance = MagicMock()
        mock_doc_gen.return_value = mock_doc_gen_instance
        mock_doc_gen_instance.generate_docx.return_value = os.path.join(self.output_dir, "test_video.docx")
        
        # Create Video2Docs instance
        converter = Video2Docs(output_dir=self.output_dir, temp_dir=self.temp_dir)
        
        # Process the video
        result = converter.process(self.video_path)
        
        # Assertions
        self.assertEqual(result, os.path.join(self.output_dir, "test_video.docx"))
        
        # Check that VideoProcessor methods were called
        mock_video_instance.extract_audio.assert_called_once_with(self.video_path)
        mock_video_instance.extract_frames.assert_called_once_with(self.video_path)
        mock_video_instance.detect_slides.assert_called_once()
        
        # Check that AudioProcessor methods were called
        mock_audio_instance.transcribe_audio.assert_called_once_with("audio.wav")
        
        # Check that LLMProcessor methods were called
        mock_llm_instance.organize_content.assert_called_once()
        
        # Check that DocumentGenerator methods were called
        mock_doc_gen_instance.generate_docx.assert_called_once()

    @patch("src.video2docs.VideoProcessor")
    @patch("src.video2docs.AudioProcessor")
    @patch("src.video2docs.LLMProcessor")
    @patch("src.video2docs.DocumentGenerator")
    def test_process_youtube_video(self, mock_doc_gen, mock_llm, mock_audio, mock_video):
        """Test processing a YouTube video."""
        # Mock VideoProcessor
        mock_video_instance = MagicMock()
        mock_video.return_value = mock_video_instance
        mock_video_instance.download_youtube_video.return_value = self.video_path
        mock_video_instance.extract_audio.return_value = "audio.wav"
        mock_video_instance.extract_frames.return_value = [
            (0.0, "frame1.jpg"),
            (1.0, "frame2.jpg")
        ]
        mock_video_instance.detect_slides.return_value = [
            (0.0, "frame1.jpg")
        ]
        
        # Mock AudioProcessor
        mock_audio_instance = MagicMock()
        mock_audio.return_value = mock_audio_instance
        mock_audio_instance.transcribe_audio.return_value = [
            {"text": "This is a test transcription.", "start_time": 0.0, "end_time": 30.0}
        ]
        
        # Mock LLMProcessor
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.organize_content.return_value = {
            "title": "Test Document",
            "summary": "This is a test summary.",
            "sections": [
                {
                    "heading": "Section 1",
                    "content": "This is section 1 content. [SLIDE 0]",
                    "bullet_points": ["Point 1", "Point 2"]
                }
            ]
        }
        
        # Mock DocumentGenerator
        mock_doc_gen_instance = MagicMock()
        mock_doc_gen.return_value = mock_doc_gen_instance
        mock_doc_gen_instance.generate_docx.return_value = os.path.join(self.output_dir, "test_video.docx")
        
        # Create Video2Docs instance
        converter = Video2Docs(output_dir=self.output_dir, temp_dir=self.temp_dir)
        
        # Process the YouTube video
        youtube_url = "https://www.youtube.com/watch?v=test"
        result = converter.process(youtube_url)
        
        # Assertions
        self.assertEqual(result, os.path.join(self.output_dir, "test_video.docx"))
        
        # Check that VideoProcessor methods were called
        mock_video_instance.download_youtube_video.assert_called_once_with(youtube_url)
        mock_video_instance.extract_audio.assert_called_once_with(self.video_path)
        mock_video_instance.extract_frames.assert_called_once_with(self.video_path)
        mock_video_instance.detect_slides.assert_called_once()
        
        # Check that AudioProcessor methods were called
        mock_audio_instance.transcribe_audio.assert_called_once_with("audio.wav")
        
        # Check that LLMProcessor methods were called
        mock_llm_instance.organize_content.assert_called_once()
        
        # Check that DocumentGenerator methods were called
        mock_doc_gen_instance.generate_docx.assert_called_once()

    @patch("src.video2docs.VideoProcessor")
    @patch("src.video2docs.AudioProcessor")
    @patch("src.video2docs.LLMProcessor")
    @patch("src.video2docs.DocumentGenerator")
    def test_process_different_formats(self, mock_doc_gen, mock_llm, mock_audio, mock_video):
        """Test processing a video to different document formats."""
        # Mock VideoProcessor
        mock_video_instance = MagicMock()
        mock_video.return_value = mock_video_instance
        mock_video_instance.extract_audio.return_value = "audio.wav"
        mock_video_instance.extract_frames.return_value = [(0.0, "frame1.jpg")]
        mock_video_instance.detect_slides.return_value = [(0.0, "frame1.jpg")]
        
        # Mock AudioProcessor
        mock_audio_instance = MagicMock()
        mock_audio.return_value = mock_audio_instance
        mock_audio_instance.transcribe_audio.return_value = [
            {"text": "This is a test transcription.", "start_time": 0.0, "end_time": 30.0}
        ]
        
        # Mock LLMProcessor
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.organize_content.return_value = {
            "title": "Test Document",
            "summary": "This is a test summary.",
            "sections": [{"heading": "Section 1", "content": "Content", "bullet_points": []}]
        }
        
        # Mock DocumentGenerator
        mock_doc_gen_instance = MagicMock()
        mock_doc_gen.return_value = mock_doc_gen_instance
        mock_doc_gen_instance.generate_docx.return_value = os.path.join(self.output_dir, "test_video.docx")
        mock_doc_gen_instance.generate_odt.return_value = os.path.join(self.output_dir, "test_video.odt")
        mock_doc_gen_instance.generate_pdf.return_value = os.path.join(self.output_dir, "test_video.pdf")
        
        # Create Video2Docs instance
        converter = Video2Docs(output_dir=self.output_dir, temp_dir=self.temp_dir)
        
        # Test DOCX format
        result_docx = converter.process(self.video_path, output_format="docx")
        self.assertEqual(result_docx, os.path.join(self.output_dir, "test_video.docx"))
        mock_doc_gen_instance.generate_docx.assert_called_once()
        
        # Test ODT format
        result_odt = converter.process(self.video_path, output_format="odt")
        self.assertEqual(result_odt, os.path.join(self.output_dir, "test_video.odt"))
        mock_doc_gen_instance.generate_odt.assert_called_once()
        
        # Test PDF format
        result_pdf = converter.process(self.video_path, output_format="pdf")
        self.assertEqual(result_pdf, os.path.join(self.output_dir, "test_video.pdf"))
        mock_doc_gen_instance.generate_pdf.assert_called_once()

    @patch("src.video2docs.VideoProcessor")
    @patch("src.video2docs.AudioProcessor")
    @patch("src.video2docs.LLMProcessor")
    @patch("src.video2docs.DocumentGenerator")
    def test_process_error_handling(self, mock_doc_gen, mock_llm, mock_audio, mock_video):
        """Test error handling during processing."""
        # Mock VideoProcessor to raise an exception
        mock_video_instance = MagicMock()
        mock_video.return_value = mock_video_instance
        mock_video_instance.extract_audio.side_effect = Exception("Audio extraction error")
        
        # Create Video2Docs instance
        converter = Video2Docs(output_dir=self.output_dir, temp_dir=self.temp_dir)
        
        # Check that the method raises the exception
        with self.assertRaises(Exception) as context:
            converter.process(self.video_path)
        
        self.assertIn("Audio extraction error", str(context.exception))


if __name__ == "__main__":
    unittest.main()