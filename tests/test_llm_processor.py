#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the LLMProcessor class.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os

from src.video2docs import LLMProcessor


class TestLLMProcessor(unittest.TestCase):
    """Test cases for the LLMProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a processor with mocked dependencies
        with patch("src.video2docs.HuggingFaceHub") as mock_hf:
            self.llm_processor = LLMProcessor(use_openai=False)
            self.mock_llm = mock_hf.return_value

    @patch("src.video2docs.LLMChain")
    def test_organize_content_success(self, mock_llm_chain):
        """Test organizing content with successful LLM response."""
        # Sample test data
        transcription = [
            {"text": "This is a test transcription.", "start_time": 0.0, "end_time": 30.0},
            {"text": "It contains multiple chunks.", "start_time": 30.0, "end_time": 60.0}
        ]
        slides = [(0.0, "slide1.jpg"), (30.0, "slide2.jpg")]
        
        # Mock LLMChain
        mock_chain = MagicMock()
        mock_llm_chain.return_value = mock_chain
        
        # Sample JSON response from LLM
        sample_response = json.dumps({
            "title": "Test Document",
            "summary": "This is a summary of the test document.",
            "sections": [
                {
                    "heading": "Introduction",
                    "content": "This is the introduction section. [SLIDE 0]",
                    "bullet_points": ["Point 1", "Point 2"]
                },
                {
                    "heading": "Main Content",
                    "content": "This is the main content section. [SLIDE 1]",
                    "bullet_points": ["Point 3", "Point 4"]
                }
            ]
        })
        
        mock_chain.run.return_value = sample_response
        
        # Call the method
        result = self.llm_processor.organize_content(transcription, slides)
        
        # Assertions
        self.assertEqual(result["title"], "Test Document")
        self.assertEqual(result["summary"], "This is a summary of the test document.")
        self.assertEqual(len(result["sections"]), 2)
        self.assertEqual(result["sections"][0]["heading"], "Introduction")
        self.assertEqual(result["sections"][1]["heading"], "Main Content")
        
        # Check that LLMChain was created and run
        mock_llm_chain.assert_called_once()
        mock_chain.run.assert_called_once()

    @patch("src.video2docs.LLMChain")
    def test_organize_content_with_markdown_code_block(self, mock_llm_chain):
        """Test organizing content with LLM response in markdown code block."""
        # Sample test data
        transcription = [
            {"text": "This is a test transcription.", "start_time": 0.0, "end_time": 30.0}
        ]
        slides = [(0.0, "slide1.jpg")]
        
        # Mock LLMChain
        mock_chain = MagicMock()
        mock_llm_chain.return_value = mock_chain
        
        # Sample response with markdown code block
        sample_response = """
        Here's the organized document:
        
        ```json
        {
            "title": "Markdown Test",
            "summary": "This is a summary with markdown.",
            "sections": [
                {
                    "heading": "Markdown Section",
                    "content": "This is content in markdown. [SLIDE 0]",
                    "bullet_points": ["Markdown Point 1", "Markdown Point 2"]
                }
            ]
        }
        ```
        """
        
        mock_chain.run.return_value = sample_response
        
        # Call the method
        result = self.llm_processor.organize_content(transcription, slides)
        
        # Assertions
        self.assertEqual(result["title"], "Markdown Test")
        self.assertEqual(result["summary"], "This is a summary with markdown.")
        self.assertEqual(len(result["sections"]), 1)
        self.assertEqual(result["sections"][0]["heading"], "Markdown Section")
        
        # Check that LLMChain was created and run
        mock_llm_chain.assert_called_once()
        mock_chain.run.assert_called_once()

    @patch("src.video2docs.LLMChain")
    def test_organize_content_parsing_error(self, mock_llm_chain):
        """Test handling parsing errors in LLM response."""
        # Sample test data
        transcription = [
            {"text": "This is a test transcription.", "start_time": 0.0, "end_time": 30.0}
        ]
        slides = [(0.0, "slide1.jpg")]
        
        # Mock LLMChain
        mock_chain = MagicMock()
        mock_llm_chain.return_value = mock_chain
        
        # Invalid JSON response
        mock_chain.run.return_value = "This is not valid JSON"
        
        # Call the method
        result = self.llm_processor.organize_content(transcription, slides)
        
        # Assertions - should return a basic structure
        self.assertEqual(result["title"], "Transcribed Video")
        self.assertTrue("summary" in result)
        self.assertEqual(len(result["sections"]), 1)
        self.assertEqual(result["sections"][0]["heading"], "Full Transcription")
        
        # Check that LLMChain was created and run
        mock_llm_chain.assert_called_once()
        mock_chain.run.assert_called_once()

    @patch("src.video2docs.OpenAI")
    @patch("src.video2docs.os.getenv")
    def test_initialize_with_openai(self, mock_getenv, mock_openai):
        """Test initializing with OpenAI."""
        # Mock environment variable
        mock_getenv.return_value = "fake-api-key"
        
        # Create processor with OpenAI
        llm_processor = LLMProcessor(use_openai=True)
        
        # Assertions
        self.assertTrue(llm_processor.use_openai)
        mock_openai.assert_called_once()

    @patch("src.video2docs.HuggingFaceHub")
    def test_initialize_with_custom_model(self, mock_hf):
        """Test initializing with custom HuggingFace model."""
        # Create processor with custom model
        llm_processor = LLMProcessor(model_name="custom/model")
        
        # Assertions
        self.assertEqual(llm_processor.model_name, "custom/model")
        mock_hf.assert_called_once()
        mock_hf.assert_called_with(
            repo_id="custom/model",
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )


if __name__ == "__main__":
    unittest.main()