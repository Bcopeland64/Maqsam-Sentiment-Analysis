# Sentiment Analysis API

This project implements a sentiment analysis API using FastAPI that serves predictions from a pre-trained multilingual transformer model optimized with ONNX Runtime. The system uses a dependency injection pattern and proper resource lifecycle management. Multiple model conversion scripts are provided, allowing flexibility in choosing the base transformer model.

## Key Components

1. **FastAPI Application** (`main.py`): Handles HTTP requests, dependency management, and serves the model predictions.
2. **Model Conversion Scripts** (e.g., `model_conversion_distilbert.py`, `model_conversion_microsoft.py`, `model_conversion_transformer.py`): Transform PyTorch transformer models from Hugging Face to ONNX format.
3. **Testing Framework** (`run_tests.py` and `test_api.py`): Provides mock environments for testing without actual model loading.

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
- `get_ml_resources()` verifies that models are loaded before endpoints use them.
- `Depends(get_ml_resources)` injects resources into endpoints that need them.
- If models aren't loaded, a 503 Service Unavailable error is returned automatically.

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
The lifespan context manager replaces older FastAPI startup/shutdown events:
- Handles model loading before the application starts serving requests.
- Yields control to the application during its lifetime.
- Handles resource cleanup on application shutdown.

### Testing Mode Detection
```python
if os.environ.get('TESTING') == 'True':
    logger.info("Testing environment detected - setting up mock model data")
    ml_models["tokenizer"] = "mock_tokenizer"
    ml_models["model"] = "mock_model"
    # ...
```
This environment-based switch:
- Creates mock model resources when `TESTING=True` is set.
- Avoids loading real models during testing.
- Provides predictable mock responses based on input text.

### Inference Pipeline
The `/predict` endpoint implements a complete NLP inference pipeline:
- Input Validation: Pydantic model validates request data.
- Tokenization: Converts text to token IDs and attention masks.
- Inference: Passes tokenized inputs to the ONNX model.
- Post-processing: Converts logits to probabilities with softmax.
- Response Formatting: Returns sentiment label, confidence, and probabilities.
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
- Uses the same dependency injection pattern to verify model readiness.
- Returns model metadata (name, label mapping).
- Confirms API operational status.

## Model Conversion

### PyTorch to ONNX Conversion
A typical conversion step involves:
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
                  # Potentially token_type_ids if the model uses them
                  'logits': {0: 'batch_size'}}
)
```
This step:
- Takes a PyTorch transformer model.
- Traces it with dummy inputs to capture the computation graph.
- Exports to ONNX format with:
  - Specified input/output names for inference.
  - Dynamic axes for variable batch sizes and sequence lengths.
  - ONNX opset version 14 (or as specified) for compatibility.
  - Constant folding for optimization.

### Verification Testing
```python
# Verification Step (Using Optimum Loader)
reloaded_model = ORTModelForSequenceClassification.from_pretrained(onnx_model_dir)
# ...
outputs = reloaded_model(**ort_inputs)
```
This verification ensures:
- The exported ONNX model loads correctly with the Optimum library.
- The label configuration (number of labels, id2label mapping) is preserved properly.
- A quick inference test passes with the expected output shape.
- Any potential conversion issues are caught immediately.

## Testing Framework

### Test Environment Setup (`run_tests.py`)
```python
# Set testing environment variable
os.environ["TESTING"] = "True"
```
This simple script:
- Sets the `TESTING` environment variable.
- Runs pytest with appropriate arguments (defaults to `test_api.py -v`).

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
- Creates an event loop to run the async lifespan context manager.
- Initializes the mock model data through the same lifespan function used by the main application.
- Ensures the application is properly set up with mock data before tests run.
- Cleans up resources when testing completes.

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
- Check multiple input scenarios including different languages.
- Verify confidence scores meet minimum thresholds based on mock logic.
- Ensure probability distributions sum to 1.0 (approximately).
- Validate the mock model's behavior without requiring real inference.

## Architecture Benefits
- **Separation of Concerns**: Model loading, API handling, and testing are clearly separated.
- **Resource Management**: Proper initialization and cleanup of ML resources via FastAPI's lifespan.
- **Testability**: Mock mode allows for rapid testing without actual model loading.
- **Error Handling**: Comprehensive exception management with appropriate HTTP status codes.
- **Performance Optimization**: ONNX Runtime for efficient inference.
- **Multilingual Support**: The chosen models can handle multiple languages, including Arabic.
- **Flexibility**: Multiple model conversion scripts allow choosing different base models.

## Setup

Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-directory>
```

Create and activate a virtual environment (Recommended):

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

Install dependencies:
```bash
pip install "fastapi[all]" uvicorn transformers torch "optimum[onnxruntime]" scipy numpy pytest sentencepiece
```

This command installs:
- FastAPI and Uvicorn (ASGI server)
- Transformers and Optimum (for ONNX integration and model handling)
- ONNX Runtime and PyTorch (needed for conversion and ONNX execution)
- Scipy/Numpy (for numerical processing)
- Pytest (for testing)
- SentencePiece (tokenizer library, often required by multilingual models)

Note: PyTorch is essential for the `model_conversion_*.py` scripts.

## Model Conversion

The API uses an ONNX model for inference. You need to run one of the provided conversion scripts first. These scripts download a pre-trained model from Hugging Face, configure it for 3 labels (Positive, Negative, Neutral), and export it to the ONNX format into the `onnx_model/` directory.

Several conversion scripts are available:
- `model_conversion_distilbert.py`: Uses `distilbert-base-multilingual-cased`.
- `model_conversion_microsoft.py`: Uses `microsoft/Multilingual-MiniLM-L12-H384`.
- `model_conversion_transformer.py`: Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Choose a script and run it. For example, to use the `distilbert-base-multilingual-cased` model:
```bash
python model_conversion_distilbert.py
```
Or, for the `microsoft/Multilingual-MiniLM-L12-H384` model:
```bash
python model_conversion_microsoft.py
```
Each script will:
- Download the specified tokenizer and model from Hugging Face.
- Configure the model with the 3 sentiment labels.
- Export the model to `model.onnx` inside the `onnx_model/` directory.
- Save the necessary tokenizer and configuration files (`config.json`, `tokenizer_config.json`, etc.) alongside the ONNX model.
- Perform a verification step to ensure the ONNX model loads correctly and produces valid outputs.

You should see output indicating the progress and success messages, ending with "✅ Conversion complete!"

## Running the Application

Once the model conversion is complete and the `onnx_model/` directory contains the `model.onnx` and associated files:

Start the Uvicorn server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- `main:app`: Tells Uvicorn to find the FastAPI instance named `app` inside the `main.py` file.
- `--reload`: Enables auto-reloading when code changes (useful for development).
- `--host 0.0.0.0`: Makes the server accessible on your network (use `127.0.0.1` for local access only).
- `--port 8000`: Specifies the port to run on.

Access the API:
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
  "sentiment": "Positive",
  "confidence": 0.987,
  "probabilities": {
    "Negative": 0.005,
    "Neutral": 0.008,
    "Positive": 0.987
  }
}
```
- `"sentiment"`: Predicted sentiment ("Positive", "Negative", or "Neutral").
- `"confidence"`: Probability of the predicted sentiment (float, 0.0 to 1.0).
- `"probabilities"`: Probability for each possible label (float).

**Error Responses:**
- 422 Unprocessable Entity: If the request body is missing the summary field or it's not a string.
- 503 Service Unavailable: If the model failed to load during startup (check application logs).
- 500 Internal Server Error: If an unexpected error occurs during processing.

### GET /health
Checks the health and readiness of the API service.

**Request Body:** None

**Success Response (200 OK):**
```json
{
  "status": "healthy",
  "model": "<model_name_from_config>",
  "ready": true,
  "labels": {
    "0": "Negative",
    "1": "Neutral",
    "2": "Positive"
  }
}
```
(The actual model name will depend on which conversion script was run and the model used.)

**Error Response (503 Service Unavailable):**
```json
{
  "detail": "Model not ready or loading failed"
}
```

## Testing

The project uses pytest for testing. Tests run quickly using mock data, simulating the model's behavior without actually loading the ONNX model.

Run tests using the helper script:
```bash
python run_tests.py
```
This script sets the `TESTING=True` environment variable and then executes `pytest test_api.py -v`.

Alternatively, you can run pytest directly if you ensure the `TESTING` environment variable is set:
```bash
# On Linux/macOS
TESTING=True pytest test_api.py -v
# On Windows (PowerShell)
$env:TESTING="True"; pytest test_api.py -v
```
(The `initialize_test_app` fixture in `test_api.py` relies on `os.environ.get('TESTING') == 'True'` being set prior to its execution for mock data loading).

What the tests cover:
- The `/health` endpoint returns a healthy status and mock model details in test mode.
- The `/predict` endpoint returns successful responses with mock data.
- The mock sentiment logic produces predictable results for certain keywords.
- Confidence scores are within the valid range [0, 1].
- Probability distributions sum to approximately 1.0.
- The highest probability in the mock response corresponds to the reported confidence score.

## Key Best Practices Implemented
- **ONNX Optimization**: Model converted to ONNX format and run with ONNX Runtime for performance.
- **Lifespan Management**: Model loaded once at startup and released at shutdown using FastAPI's lifespan context manager.
- **Dependency Injection**: `Depends` used to provide ML resources to endpoints, ensuring they are ready.
- **Environment-Aware Configuration**: `TESTING` environment variable controls loading mock data vs. the real model.
- **Input Validation**: Pydantic models (e.g., `SummaryRequest`) automatically validate incoming request data.
- **Structured Logging**: Consistent log format aids monitoring and debugging.
- **Global Exception Handling**: Catches unhandled errors and returns standardized 500 responses.
- **Health Check Endpoint**: Provides a standard `/health` route for monitoring and orchestration.
- **Testability**: Mock data mechanism allows for fast, isolated testing suitable for CI/CD pipelines.
- **Resource Protection**: Tokenizer truncation (e.g., max_length) helps prevent excessive resource usage from overly long inputs.
