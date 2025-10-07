#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test runner for Video2Docs tests.
"""

import unittest
import sys
import os

# Add the parent directory to the path so that the tests can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from tests.test_video_processor import TestVideoProcessor
from tests.test_audio_processor import TestAudioProcessor
from tests.test_llm_processor import TestLLMProcessor
from tests.test_document_generator import TestDocumentGenerator
from tests.test_video2docs import TestVideo2Docs


def run_tests():
    """Run all tests."""
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add test cases using the modern, supported loader (makeSuite is removed in Python 3.13)
    loader = unittest.defaultTestLoader
    test_suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessor))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAudioProcessor))
    test_suite.addTests(loader.loadTestsFromTestCase(TestLLMProcessor))
    test_suite.addTests(loader.loadTestsFromTestCase(TestDocumentGenerator))
    test_suite.addTests(loader.loadTestsFromTestCase(TestVideo2Docs))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return the result
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)