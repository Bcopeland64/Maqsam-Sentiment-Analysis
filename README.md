# Sentiment Analysis API Documentation

## System Overview

This project implements a sentiment analysis API using FastAPI that serves predictions from a pre-trained multilingual transformer model optimized with ONNX Runtime. The system uses a dependency injection pattern and proper resource lifecycle management.

## Key Components

1. **FastAPI Application** (`main.py`): Handles HTTP requests, dependency management, and serves the model predictions
2. **Model Conversion Script** (`model_conversion.py`): Transforms a PyTorch transformer model to ONNX format
3. **Testing Framework** (`run_tests.py` and `test_api.py`): Provides mock environments for testing without actual model loading

## Deep Dive: FastAPI Application (`main.py`)

### Dependency Injection Pattern

```python
# Dependency Function
async def get_ml_resources():
    if not ml_models:  # Check if the dictionary is empty (loading failed)
        raise HTTPException(status_code=503, detail="Model not ready or loading failed")
    return ml_models

# Injection in endpoint
@app.post("/predict")
async def predict(
    request: SummaryRequest,
    # Inject the resources using Depends
    resources: dict = Depends(get_ml_resources)
):
```

This pattern separates resource management from endpoint logic:

- `get_ml_resources()` verifies that models are loaded before endpoints use them
- `Depends(get_ml_resources)` injects resources into endpoints that need them
- If models aren't loaded, a 503 Service Unavailable error is returned automatically

### Lifespan Management

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup code - loads models or mocks for testing
    
    yield  # Application runs here
    
    # Shutdown: Clear the resources
    logger.info("Application shutdown: Clearing model resources...")
    ml_models.clear()
```

The `lifespan` context manager replaces older FastAPI startup/shutdown events:

1. Handles model loading before the application starts serving requests
2. Yields control to the application during its lifetime
3. Handles resource cleanup on application shutdown

### Testing Mode Detection

```python
if os.environ.get('TESTING') == 'True':
    logger.info("Testing environment detected - setting up mock model data")
    ml_models["tokenizer"] = "mock_tokenizer"
    ml_models["model"] = "mock_model"
    # ... 
```

This environment-based switch:

- Creates mock model resources when `TESTING=True` is set
- Avoids loading real models during testing
- Provides predictable mock responses based on input text

### Inference Pipeline

The `/predict` endpoint implements a complete NLP inference pipeline:

1. **Input Validation**: Pydantic model validates request data
2. **Tokenization**: Converts text to token IDs and attention masks
3. **Inference**: Passes tokenized inputs to the ONNX model
4. **Post-processing**: Converts logits to probabilities with softmax
5. **Response Formatting**: Returns sentiment label, confidence, and probabilities

Error handling wraps each step with specific error messages and status codes.

### Health Check Endpoint

```python
@app.get("/health")
async def health_check(
    resources: dict = Depends(get_ml_resources)
):
    # If Depends(get_ml_resources) succeeded, the model is loaded.
    # ...
```

The health check endpoint:

- Uses the same dependency injection pattern to verify model readiness
- Returns model metadata (name, label mapping)
- Confirms API operational status

## Deep Dive: Model Conversion (`model_conversion.py`)

### PyTorch to ONNX Conversion

```python
torch.onnx.export(
    pt_model,
    tuple(dummy_inputs.values()),
    onnx_model_path,
    export_params=True,
    opset_version=14,             # Use opset 14 or higher
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                  'attention_mask': {0: 'batch_size', 1: 'sequence'},
                  'logits': {0: 'batch_size'}}
)
```

This step:

1. Takes a PyTorch transformer model 
2. Traces it with dummy inputs to capture the computation graph
3. Exports to ONNX format with:
   - Specified input/output names for inference
   - Dynamic axes for variable batch sizes and sequence lengths
   - ONNX opset version 14 for compatibility
   - Constant folding for optimization

### Verification Testing

```python
# Verification Step (Using Optimum Loader)
reloaded_model = ORTModelForSequenceClassification.from_pretrained(onnx_model_dir)
# ...
outputs = reloaded_model(**ort_inputs)
```

This verification ensures:

1. The exported ONNX model loads correctly with the Optimum library
2. The label configuration is preserved properly
3. A quick inference test passes with the expected output shape
4. Any potential conversion issues are caught immediately

## Deep Dive: Testing Framework

### Test Environment Setup (`run_tests.py`)

```python
# Set testing environment variable
os.environ["TESTING"] = "True"
```

This simple script:
1. Sets the `TESTING` environment variable
2. Runs pytest with appropriate arguments

### Test Client and Lifespan Management (`test_api.py`)

```python
@pytest.fixture(scope="session", autouse=True)
def initialize_test_app():
    async def init_models():
        async with lifespan(app):
            yield
    
    loop = asyncio.new_event_loop()
    try:
        gen = init_models().__aiter__()
        loop.run_until_complete(gen.__anext__())
        yield
    finally:
        loop.close()
```

This fixture:

1. Creates an event loop to run the async lifespan context manager
2. Initializes the mock model data through the same lifespan function
3. Ensures the application is properly set up before tests run
4. Cleans up resources when testing completes

### Parametrized Tests

```python
@pytest.mark.parametrize("input_text,min_confidence", [
    ("I'm furious about this terrible service!", 0.35),
    ("The product is okay I guess", 0.35),
    # ... including Arabic text tests
])
def test_sentiment_analysis(input_text, min_confidence):
    # ...
```

These tests:

1. Check multiple input scenarios including different languages
2. Verify confidence scores meet minimum thresholds
3. Ensure probability distributions sum to 1.0
4. Validate the mock model's behavior without requiring real inference

## Architecture Benefits

1. **Separation of Concerns**: Model loading, API handling, and testing are clearly separated
2. **Resource Management**: Proper initialization and cleanup of ML resources
3. **Testability**: Mock mode allows testing without actual model loading
4. **Error Handling**: Comprehensive exception management with appropriate HTTP status codes
5. **Performance Optimization**: ONNX Runtime for efficient inference
6. **Multilingual Support**: The model handles multiple languages including Arabic

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment (Recommended):**
   
   Using venv:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
   
   Using conda:
   ```bash
   conda create -n sentiment_env python=3.9  # Or your preferred Python version
   conda activate sentiment_env
   ```

3. **Install dependencies:**
   ```bash
   pip install "fastapi[all]" uvicorn transformers torch "optimum[onnxruntime]" scipy numpy pytest
   ```
   
   This command installs:
   - FastAPI and Uvicorn (ASGI server)
   - Transformers and Optimum (for ONNX integration)
   - ONNX Runtime and PyTorch (needed for conversion)
   - Scipy/Numpy (for processing)
   - Pytest (for testing)
   
   Note: You might not need torch after conversion if only running the API, but it's required for model_conversion.py

## Model Conversion

The API uses an ONNX model for inference. You need to run the conversion script first, which downloads the pre-trained `distilbert-base-multilingual-cased` model from Hugging Face, configures it for 3 labels (Positive, Negative, Neutral), and exports it to the ONNX format.

**Run the conversion script:**
```bash
python model_conversion.py
```

This script will:
- Download the tokenizer and model from Hugging Face
- Configure the model with the specified labels
- Export the model to `model.onnx` inside the `onnx_model/` directory
- Save the necessary tokenizer and configuration files alongside the ONNX model
- Perform a verification step to ensure the ONNX model loads correctly

You should see output indicating the progress and success messages, ending with "✅ Conversion complete!"

## Running the Application

Once the model conversion is complete and the `onnx_model/` directory exists:

1. **Start the Uvicorn server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   - `main:app`: Tells Uvicorn to find the FastAPI instance named `app` inside the `main.py` file
   - `--reload`: Enables auto-reloading when code changes (useful for development)
   - `--host 0.0.0.0`: Makes the server accessible on your network (use `127.0.0.1` for local access only)
   - `--port 8000`: Specifies the port to run on

2. **Access the API:**
   - The API will be running at http://127.0.0.1:8000
   - Interactive API documentation (Swagger UI): http://127.0.0.1:8000/docs
   - Alternative API documentation (ReDoc): http://127.0.0.1:8000/redoc

## API Endpoints

### POST /predict

Analyzes the sentiment of the provided text.

**Request Body:**
```json
{
  "summary": "Your text to analyze here. Este es un texto en español."
}
```

- `summary` (string, required): The text input for sentiment analysis.

**Success Response (200 OK):**
```json
{
  "sentiment": "Positive",  // "Positive", "Negative", or "Neutral"
  "confidence": 0.987,      // Probability of the predicted sentiment (float, 0.0 to 1.0)
  "probabilities": {
    "Negative": 0.005,
    "Neutral": 0.008,
    "Positive": 0.987     // Probability for each possible label (float)
  }
}
```

**Error Responses:**
- 422 Unprocessable Entity: If the request body is missing the summary field or it's not a string
- 503 Service Unavailable: If the model failed to load during startup (check application logs)
- 500 Internal Server Error: If an unexpected error occurs during processing

### GET /health

Checks the health and readiness of the API service.

**Request Body:** None

**Success Response (200 OK):**
```json
{
  "status": "healthy",
  "model": "distilbert-base-multilingual-cased",  // Or the model name from config
  "ready": true,
  "labels": {  // Label mapping used by the model
    "0": "Negative",
    "1": "Neutral",
    "2": "Positive"
  }
}
```

**Error Response (503 Service Unavailable):**
```json
{
  "detail": "Model not ready or loading failed"
}
```

## Testing

The project uses pytest for testing. Tests run quickly using mock data, simulating the model's behavior without actually loading the ONNX model.

1. **Run tests using the helper script:**
   ```bash
   python run_tests.py
   ```
   
   Or run pytest directly:
   ```bash
   # TESTING=True pytest test_api.py -v  # If running manually
   pytest test_api.py -v  # The fixture in test_api.py handles setting TESTING=True
   ```

2. **What the tests cover:**
   - The `/health` endpoint returns a healthy status in test mode
   - The `/predict` endpoint returns successful responses
   - The mock sentiment logic produces expected results for certain keywords
   - Confidence scores are within the valid range [0, 1]
   - Probability distributions sum to approximately 1.0
   - The highest probability corresponds to the reported confidence score

## Key Best Practices Implemented

- **ONNX Optimization**: Model converted to ONNX format and run with ONNX Runtime for performance
- **Lifespan Management**: Model loaded once at startup and released at shutdown using FastAPI's lifespan context manager
- **Dependency Injection**: Depends used to provide ML resources to endpoints, ensuring they are ready
- **Environment-Aware Configuration**: TESTING environment variable controls loading mock data vs. the real model
- **Input Validation**: Pydantic models (SummaryRequest) automatically validate incoming request data
- **Structured Logging**: Consistent log format aids monitoring and debugging
- **Global Exception Handling**: Catches unhandled errors and returns standardized 500 responses
- **Health Check Endpoint**: Provides a standard /health route for monitoring and orchestration
- **Testability**: Mock data mechanism allows for fast, isolated testing suitable for CI/CD pipelines
- **Resource Protection**: Tokenizer truncation (max_length) prevents excessive resource usage from overly long inputs
