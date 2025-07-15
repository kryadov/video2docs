#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video2Docs Example Script

This script demonstrates how to use the Video2Docs application to convert
a YouTube video or local video file to a document.
"""

import os
import sys
from video2docs import Video2Docs

def youtube_example():
    """Example of converting a YouTube video to a document."""
    print("Example 1: Converting a YouTube video to a DOCX document")

    # Initialize the converter
    converter = Video2Docs(
        output_dir="output",
        use_gpu=True
    )

    # YouTube URL (TED Talk example)
    youtube_url = "https://www.youtube.com/watch?v=8S0FDjFBj8o"  # Sample TED Talk

    # Convert to DOCX
    try:
        output_path = converter.process(youtube_url, output_format="docx")
        print(f"Success! Document generated at: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        return False

    return True

def local_video_example(video_path):
    """Example of converting a local video file to a document."""
    print(f"Example 2: Converting a local video file to a PDF document")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False

    # Initialize the converter
    converter = Video2Docs(
        output_dir="output",
        use_gpu=True
    )

    # Convert to PDF
    try:
        output_path = converter.process(video_path, output_format="pdf")
        print(f"Success! Document generated at: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        return False

    return True

def main():
    """Run the examples."""
    print("Video2Docs Examples")
    print("===================")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Example 1: YouTube video
    success = youtube_example()
    print()

    # Example 2: Local video file
    # Replace with your own video file path or use a sample video
    sample_video_path = "path/to/your/video.mp4"
    if len(sys.argv) > 1:
        sample_video_path = sys.argv[1]

    if os.path.exists(sample_video_path):
        local_video_example(sample_video_path)
    else:
        print(f"Skipping local video example. To run it, provide a video file path as an argument:")
        print(f"python example.py path/to/your/video.mp4")

    print("\nExamples completed.")

if __name__ == "__main__":
    main()
