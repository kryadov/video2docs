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

- Python 3.10+ (3.11 recommended)
- FFmpeg installed and on PATH (required by MoviePy and pydub)
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone this repository:
   ```shell
   git clone https://github.com/yourusername/video2docs.git
   cd video2docs
   ```

2. Create a virtual environment (recommended):
   ```shell
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```PowerShell
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```shell
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```shell
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
   - For YouTube videos: Downloads the video using pytubefix
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

## API Usage and Costs

When using OpenAI's ChatGPT API for content organization, it's important to understand how video length affects API usage and costs:

- **Token Usage**: The application sends the first 1500 characters of the transcribed text to the LLM. This typically amounts to approximately 300-500 tokens, depending on the complexity of the language.

- **Cost Estimation by Video Length**:
  - **Short videos (1-5 minutes)**: Approximately 500-1500 tokens, costing around $0.01-$0.03 with GPT-3.5 or $0.05-$0.15 with GPT-4.
  - **Medium videos (5-15 minutes)**: The transcription may exceed 1500 characters, but the application still only sends the first 1500 characters, keeping costs similar to short videos.
  - **Long videos (15+ minutes)**: Despite the length, API costs remain constant since only the first 1500 characters are processed. However, this may result in incomplete content organization for very long videos.

- **Optimizing Costs**:
  - For longer videos, consider breaking them into smaller segments for better content organization.
  - Use the Hugging Face models option instead of OpenAI for cost-free processing (though quality may vary).
  - Adjust the character limit in the source code if you need more comprehensive processing of longer videos (this will increase API costs).

Note: These estimates are based on OpenAI's pricing as of 2023. Check [OpenAI's pricing page](https://openai.com/pricing) for the most current rates.

## Customization

You can customize the behavior of the application by modifying the parameters in the `src/video2docs.py` file:

- Change the default LLM model by modifying the `model_name` parameter in the `LLMProcessor` class
- Adjust the frame extraction interval by modifying the `interval` parameter in the `extract_frames` method
- Modify the slide detection threshold by changing the `threshold` parameter in the `detect_slides` method

## Troubleshooting

- **GPU not being used**: Make sure you have the appropriate CUDA drivers installed and that PyTorch can detect your GPU. You can check this by running `torch.cuda.is_available()` in a Python shell.
- **Memory errors**: Processing large videos can be memory-intensive. Try reducing the frame extraction interval or processing shorter video segments.
- **Missing dependencies**: Make sure all dependencies are installed by running `pip install -r requirements.txt`.

## Web UI (Beta)

A simple Bootstrap-powered web UI is included.

1. Set up your environment in a `.env` file (optional, defaults shown):
   ```env
   ADMIN_USER=admin
   ADMIN_PASS=admin123
   SECRET_KEY=change-me
   OUTPUT_DIR=output
   # TEMP_DIR=output\\temp
   ```
2. Install requirements:
   ```shell
   pip install -r requirements.txt
   ```
3. Run the web app:
   ```shell
   python -m src.webapp
   ```
   Alternatively, on Windows you can run run_web.bat; on macOS/Linux, run run_web.sh
4. Open http://localhost:5000 and log in with the configured credentials.

Features:
- Login (credentials from .env)
- New conversion: YouTube URL or file upload, choose format, optional output filename
- History list with details page per conversion
- Ability to re-run a previous conversion with modified parameters
- Download the generated document from the detail page

Notes:
- Conversions and their metadata are stored in `OUTPUT_DIR/conversions.json`.
- Uploaded files are saved under `OUTPUT_DIR/uploads`.
- Processing runs in the background with live progress and ETA. You can cancel a running conversion from the dedicated Running page. Closing the browser/tab will not interrupt the process.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
