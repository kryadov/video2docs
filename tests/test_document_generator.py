#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the DocumentGenerator class.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile

from src.video2docs import DocumentGenerator


class TestDocumentGenerator(unittest.TestCase):
    """Test cases for the DocumentGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.document_generator = DocumentGenerator(output_dir=self.temp_dir)
        
        # Sample content for testing
        self.content = {
            "title": "Test Document",
            "summary": "This is a test summary.",
            "sections": [
                {
                    "heading": "Section 1",
                    "content": "This is section 1 content. [SLIDE 0]",
                    "bullet_points": ["Point 1", "Point 2"]
                },
                {
                    "heading": "Section 2",
                    "content": "This is section 2 content. [SLIDE 1]",
                    "bullet_points": ["Point 3", "Point 4"]
                }
            ]
        }
        
        # Sample slides for testing
        self.slides = [
            (0.0, os.path.join(self.temp_dir, "slide1.jpg")),
            (30.0, os.path.join(self.temp_dir, "slide2.jpg"))
        ]
        
        # Create sample slide files
        for _, slide_path in self.slides:
            with open(slide_path, "w") as f:
                f.write("mock image data")

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        for _, slide_path in self.slides:
            if os.path.exists(slide_path):
                os.remove(slide_path)
        
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    @patch("src.video2docs.docx.Document")
    def test_generate_docx(self, mock_document_class):
        """Test generating a DOCX document."""
        # Mock Document class and its methods
        mock_document = MagicMock()
        mock_document_class.return_value = mock_document
        
        # Mock document elements
        mock_paragraph = MagicMock()
        mock_document.add_paragraph.return_value = mock_paragraph
        
        # Output path
        output_path = os.path.join(self.temp_dir, "test.docx")
        
        # Call the method
        result = self.document_generator.generate_docx(self.content, self.slides, output_path)
        
        # Assertions
        self.assertEqual(result, output_path)
        
        # Check that Document was created
        mock_document_class.assert_called_once()
        
        # Check that title was added
        mock_document.add_heading.assert_any_call(self.content["title"], level=0)
        
        # Check that summary was added
        mock_document.add_heading.assert_any_call("Executive Summary", level=1)
        mock_document.add_paragraph.assert_any_call(self.content["summary"])
        
        # Check that sections were added
        mock_document.add_heading.assert_any_call("Section 1", level=1)
        mock_document.add_heading.assert_any_call("Section 2", level=1)
        
        # Check that document was saved
        mock_document.save.assert_called_once_with(output_path)

    @patch("src.video2docs.OpenDocumentText")
    def test_generate_odt(self, mock_odt_class):
        """Test generating an ODT document."""
        # Mock OpenDocumentText class and its methods
        mock_document = MagicMock()
        mock_odt_class.return_value = mock_document
        
        # Mock document elements
        mock_text = MagicMock()
        mock_document.text = mock_text
        
        mock_styles = MagicMock()
        mock_document.styles = mock_styles
        
        # Output path
        output_path = os.path.join(self.temp_dir, "test.odt")
        
        # Call the method
        result = self.document_generator.generate_odt(self.content, self.slides, output_path)
        
        # Assertions
        self.assertEqual(result, output_path)
        
        # Check that OpenDocumentText was created
        mock_odt_class.assert_called_once()
        
        # Check that styles were added
        mock_styles.addElement.assert_called_once()
        
        # Check that document was saved
        mock_document.save.assert_called_once_with(output_path)

    @patch("src.video2docs.FPDF")
    def test_generate_pdf(self, mock_fpdf_class):
        """Test generating a PDF document."""
        # Mock FPDF class and its methods
        mock_pdf = MagicMock()
        mock_fpdf_class.return_value = mock_pdf
        
        # Output path
        output_path = os.path.join(self.temp_dir, "test.pdf")
        
        # Call the method
        result = self.document_generator.generate_pdf(self.content, self.slides, output_path)
        
        # Assertions
        self.assertEqual(result, output_path)
        
        # Check that FPDF was created
        mock_fpdf_class.assert_called_once()
        
        # Check that page was added
        mock_pdf.add_page.assert_called_once()
        
        # Check that font was set
        mock_pdf.set_font.assert_called()
        
        # Check that title was added
        mock_pdf.cell.assert_any_call(0, 10, self.content["title"], ln=True, align="C")
        
        # Check that document was saved
        mock_pdf.output.assert_called_once_with(output_path)

    def test_init_creates_output_dir(self):
        """Test that initializing creates the output directory."""
        # Create a new temporary path that doesn't exist
        new_dir = os.path.join(self.temp_dir, "new_output_dir")
        
        # Initialize with the new directory
        DocumentGenerator(output_dir=new_dir)
        
        # Check that the directory was created
        self.assertTrue(os.path.exists(new_dir))
        
        # Clean up
        os.rmdir(new_dir)


if __name__ == "__main__":
    unittest.main()