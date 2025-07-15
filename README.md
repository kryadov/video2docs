# Video2Docs

A Python application that converts YouTube videos or local video files to document formats (ODT, DOCX, PDF). The application extracts text from speech, identifies slides/images, and uses LLMs to organize the content into a well-structured document.

## Features

- Convert YouTube videos to documents by providing a URL
- Convert local video files to documents
- Automatically extract text from speech using speech recognition
- Detect and extract slides/images from the video
- Use LLMs to organize content into a structured document
- Generate documents in multiple formats (ODT, DOCX, PDF)
- GPU acceleration for faster processing (when available)

## Requirements

- Python 3.13 or higher
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/video2docs.git
   cd video2docs
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

5. (Optional) For GPU acceleration, make sure you have the appropriate CUDA drivers installed for your GPU.

6. (Optional) For OpenAI integration, create a `.env` file in the project root and add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   ```

## Usage

### Basic Usage

Convert a YouTube video to a DOCX document:

```
python src/video2docs.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Convert a local video file to a PDF document:

```
python src/video2docs.py path/to/your/video.mp4 --format pdf
```

### Command-line Options

- `input`: YouTube URL or path to local video file (required)
- `--format`, `-f`: Output document format (choices: "docx", "odt", "pdf", default: "docx")
- `--output-dir`, `-o`: Directory to save output documents (default: "output")
- `--temp-dir`, `-t`: Directory for temporary files (default: auto-generated)
- `--no-gpu`: Disable GPU usage
- `--verbose`, `-v`: Enable verbose logging

### Examples

Convert a YouTube video to a DOCX document with verbose logging:

```
python src/video2docs.py "https://www.youtube.com/watch?v=VIDEO_ID" -v
```

Convert a local video file to an ODT document and save it in a specific directory:

```
python src/video2docs.py path/to/your/video.mp4 --format odt --output-dir my_documents
```

## How It Works

1. **Video Processing**:
   - For YouTube videos: Downloads the video using pytube
   - For local files: Uses the provided file path
   - Extracts frames from the video at regular intervals

2. **Audio Processing**:
   - Extracts audio from the video
   - Transcribes the audio to text using speech recognition

3. **Slide Detection**:
   - Analyzes extracted frames to detect slides/images
   - Uses structural similarity to identify unique slides

4. **Content Organization**:
   - Uses LLMs (either Hugging Face models or OpenAI) to organize the transcribed text
   - Creates a structured document with sections, bullet points, and slide placements

5. **Document Generation**:
   - Generates the document in the specified format (DOCX, ODT, or PDF)
   - Includes detected slides/images in the appropriate locations

## Customization

You can customize the behavior of the application by modifying the parameters in the `src/video2docs.py` file:

- Change the default LLM model by modifying the `model_name` parameter in the `LLMProcessor` class
- Adjust the frame extraction interval by modifying the `interval` parameter in the `extract_frames` method
- Modify the slide detection threshold by changing the `threshold` parameter in the `detect_slides` method

## Troubleshooting

- **GPU not being used**: Make sure you have the appropriate CUDA drivers installed and that PyTorch can detect your GPU. You can check this by running `torch.cuda.is_available()` in a Python shell.
- **Memory errors**: Processing large videos can be memory-intensive. Try reducing the frame extraction interval or processing shorter video segments.
- **Missing dependencies**: Make sure all dependencies are installed by running `pip install -r requirements.txt`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.