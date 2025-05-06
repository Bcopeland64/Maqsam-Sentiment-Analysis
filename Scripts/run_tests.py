#!/usr/bin/env python3
import os
import sys
import pytest

if __name__ == "__main__":
    # Set testing environment variable
    os.environ["TESTING"] = "True"
    
    # Print testing information
    print("Running tests in TESTING mode with mock model data")
    
    # Run pytest with any arguments passed to this script
    args = sys.argv[1:] or ["test_api.py", "-v"]
    sys.exit(pytest.main(args))