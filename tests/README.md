# Video2Docs Tests

This directory contains tests for the Video2Docs application. The tests are organized by component and include both unit tests and integration tests.

## Test Structure

- `test_video_processor.py`: Tests for the VideoProcessor class, which handles video downloading and processing.
- `test_audio_processor.py`: Tests for the AudioProcessor class, which handles audio transcription.
- `test_llm_processor.py`: Tests for the LLMProcessor class, which handles LLM processing for content organization.
- `test_document_generator.py`: Tests for the DocumentGenerator class, which generates documents in various formats.
- `test_video2docs.py`: Integration tests for the Video2Docs class, which integrates all the components.
- `run_tests.py`: Script to run all tests together.

## Prerequisites

Before running the tests, make sure you have installed all the required dependencies:

```bash
pip install -r requirements.txt
```

If you get "GPU is not available, using CPU" in the log then run
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```


The tests use mocking to avoid relying on actual external services, but the imports from the main application still need to be available.

## Running Tests

### Running All Tests

To run all tests, use the `run_tests.py` script:

```bash
python tests/run_tests.py
```

### Running Individual Test Files

To run tests for a specific component, use the unittest module:

```bash
python -m unittest tests.test_video_processor
python -m unittest tests.test_audio_processor
python -m unittest tests.test_llm_processor
python -m unittest tests.test_document_generator
python -m unittest tests.test_video2docs
```

### Running Specific Test Cases

To run a specific test case, use the unittest module with the test case name:

```bash
python -m unittest tests.test_video_processor.TestVideoProcessor.test_download_youtube_video
```

## Test Coverage

The tests cover the following functionality:

- **VideoProcessor**:
  - Downloading YouTube videos
  - Extracting audio from video files
  - Extracting frames from video files
  - Detecting slides in frames

- **AudioProcessor**:
  - Transcribing audio to text
  - Handling empty audio
  - Error handling during transcription

- **LLMProcessor**:
  - Organizing content with successful LLM response
  - Handling LLM responses in markdown code blocks
  - Handling parsing errors in LLM responses
  - Initializing with OpenAI or HuggingFace models

- **DocumentGenerator**:
  - Generating DOCX documents
  - Generating ODT documents
  - Generating PDF documents
  - Creating output directories

- **Video2Docs (Integration)**:
  - Processing local video files
  - Processing YouTube videos
  - Processing videos to different document formats
  - Error handling during processing

## Adding New Tests

When adding new functionality to the Video2Docs application, make sure to add corresponding tests. Follow these guidelines:

1. Add unit tests for new methods in the appropriate test file.
2. Use mocking to avoid relying on external services or actual files.
3. Add integration tests for new workflows in the `test_video2docs.py` file.
4. Update the `run_tests.py` script if necessary.

## Test Dependencies

The tests use the following Python modules:

- `unittest`: Standard Python testing framework
- `unittest.mock`: For mocking external dependencies
- `tempfile`: For creating temporary files and directories
- `shutil`: For cleaning up temporary files and directories
