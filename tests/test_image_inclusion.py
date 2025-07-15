#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify that images are included in generated documents.
"""

import os
import tempfile
import shutil
from PIL import Image
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.video2docs import DocumentGenerator

def create_test_image(path, color=(255, 0, 0)):
    """Create a simple test image."""
    # Create a 100x100 red image
    img = Image.new('RGB', (100, 100), color=color)
    img.save(path)
    return path

def main():
    """Main function to test image inclusion in documents."""
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create test images
        slide1_path = os.path.join(temp_dir, "slide1.jpg")
        slide2_path = os.path.join(temp_dir, "slide2.jpg")

        create_test_image(slide1_path, color=(255, 0, 0))  # Red
        create_test_image(slide2_path, color=(0, 0, 255))  # Blue

        # Sample slides
        slides = [
            (0.0, slide1_path),
            (30.0, slide2_path)
        ]

        # Sample content
        content = {
            "title": "Test Document with Images",
            "summary": "This document tests the inclusion of images.",
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

        # Initialize document generator
        document_generator = DocumentGenerator(output_dir=output_dir)

        # Generate documents in all formats
        docx_path = document_generator.generate_docx(content, slides, os.path.join(output_dir, "test_images.docx"))
        odt_path = document_generator.generate_odt(content, slides, os.path.join(output_dir, "test_images.odt"))
        pdf_path = document_generator.generate_pdf(content, slides, os.path.join(output_dir, "test_images.pdf"))

        print(f"Generated documents:")
        print(f"DOCX: {docx_path}")
        print(f"ODT: {odt_path}")
        print(f"PDF: {pdf_path}")
        print(f"\nPlease open these documents to verify that images are included.")

    finally:
        # Don't clean up so we can examine the files
        print(f"\nTemporary files are in: {temp_dir}")
        print("Please delete this directory manually when done.")

if __name__ == "__main__":
    main()
