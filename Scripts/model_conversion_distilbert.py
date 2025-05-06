# model_conversion.py (Updated opset_version to 14)
import torch
import os
import shutil
import logging
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
# Need ORTModel class for verification loading
from optimum.onnxruntime import ORTModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
model_name = "distilbert-base-multilingual-cased"
num_labels = 3
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
label2id = {v: k for k, v in id2label.items()}
onnx_model_dir = "onnx_model"
onnx_model_filename = "model.onnx" # Name of the ONNX file
# --- End Configuration ---

# Clean up previous model
if os.path.exists(onnx_model_dir):
    logger.info(f"Removing existing directory: {onnx_model_dir}")
    shutil.rmtree(onnx_model_dir)
os.makedirs(onnx_model_dir, exist_ok=True)

logger.info(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

logger.info(f"Loading config for {model_name} and setting labels...")
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

logger.info(f"Loading PyTorch model '{model_name}' with {num_labels} labels...")
pt_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config
)
pt_model.eval()
logger.info("PyTorch model loaded successfully.")

# --- Direct PyTorch ONNX Export ---
logger.info("Performing direct PyTorch ONNX export...")
onnx_model_path = os.path.join(onnx_model_dir, onnx_model_filename)

try:
    dummy_text = "Some dummy input text for tracing."
    # Use max_length the same as your API expects, or a reasonable default like 128/256
    dummy_inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=150)
    input_names = list(dummy_inputs.keys())
    output_names = ['logits']

    dummy_inputs = {k: v.to('cpu') for k, v in dummy_inputs.items()}
    pt_model.to('cpu')

    logger.info(f"Input names: {input_names}")
    logger.info(f"Output names: {output_names}")
    logger.info(f"Input shapes: {[v.shape for v in dummy_inputs.values()]}")

    with torch.no_grad():
        torch.onnx.export(
            pt_model,
            tuple(dummy_inputs.values()),
            onnx_model_path,
            export_params=True,
            # *** UPDATED OPSET VERSION HERE ***
            opset_version=14,             # Use opset 14 or higher
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                          'attention_mask': {0: 'batch_size', 1: 'sequence'},
                          'logits': {0: 'batch_size'}}
        )
    logger.info(f"Direct PyTorch ONNX export successful to {onnx_model_path} using opset 14")

except Exception as e:
    logger.exception("🚨 Error during direct PyTorch ONNX export!")
    raise e

# --- Save Config and Tokenizer ---
logger.info(f"Saving config and tokenizer to {onnx_model_dir}...")
config.save_pretrained(onnx_model_dir)     # Save the config.json with label info
tokenizer.save_pretrained(onnx_model_dir) # Save tokenizer files

# --- Verification Step (Using Optimum Loader) ---
try:
    logger.info("Verifying the saved ONNX model using ORTModelForSequenceClassification...")
    reloaded_model = ORTModelForSequenceClassification.from_pretrained(onnx_model_dir)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(onnx_model_dir)
    logger.info("Verification load successful.")

    logger.info(f"Loaded model config name/path: {getattr(reloaded_model.config, '_name_or_path', 'N/A')}")
    logger.info(f"Loaded model config labels: {reloaded_model.config.num_labels}")
    logger.info(f"Loaded model config id2label: {reloaded_model.config.id2label}")
    loaded_id2label = {int(k): v for k,v in reloaded_model.config.id2label.items()}
    assert reloaded_model.config.num_labels == num_labels, f"Expected {num_labels} labels, found {reloaded_model.config.num_labels}"
    assert loaded_id2label == id2label, f"Expected id2label {id2label}, found {loaded_id2label}"

    logger.info("Performing quick inference test...")
    test_text = "This is a verification test."
    inputs_pt = reloaded_tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=150) # Match dummy max_length
    ort_inputs = {k: v.cpu().numpy() for k, v in inputs_pt.items()}

    outputs = reloaded_model(**ort_inputs)
    logger.info(f"Inference test output logits shape: {outputs.logits.shape}")

    assert isinstance(outputs.logits, np.ndarray), "Output logits should be a numpy array"
    assert outputs.logits.shape[0] == 1, f"Expected batch size 1, found {outputs.logits.shape[0]}"
    assert outputs.logits.shape[-1] == num_labels, f"Expected output dim {num_labels}, found {outputs.logits.shape[-1]}"
    logger.info("Inference test successful.")

except Exception as e:
    logger.exception("🚨 Error during ONNX model verification!")
    raise e
# --- End Verification Step ---

logger.info(f"✅ Conversion complete! Model saved to {onnx_model_dir}/")