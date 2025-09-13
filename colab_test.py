"""
Quick test script for Google Colab - Tests GPU and loads a smaller model first
Run this to verify your setup before loading the full model
"""

# Test 1: Check GPU
import torch
print("=" * 50)
print("🔍 CHECKING ENVIRONMENT")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ NO GPU DETECTED! Please enable GPU:")
    print("   Runtime > Change runtime type > Hardware accelerator > GPU (T4)")
    exit()

# Test 2: Install requirements
print("\n" + "=" * 50)
print("📦 INSTALLING PACKAGES")
print("=" * 50)
import subprocess
import sys

packages = ["gradio", "transformers", "accelerate", "bitsandbytes"]
for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
print("✅ All packages installed")

# Test 3: Test with smaller model first
print("\n" + "=" * 50)
print("🧪 TESTING WITH SMALL MODEL FIRST")
print("=" * 50)

from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Use a tiny model for testing
test_model = "microsoft/DialoGPT-small"
print(f"Loading test model: {test_model}")

tokenizer = AutoTokenizer.from_pretrained(test_model)
model = AutoModelForCausalLM.from_pretrained(test_model).to("cuda")
print("✅ Test model loaded successfully!")

# Simple test generation
test_input = "Hello, how are you?"
inputs = tokenizer.encode(test_input, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nTest Input: {test_input}")
print(f"Test Output: {response}")

# Clean up
del model
torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED!")
print("=" * 50)
print("\nYour environment is ready! Now you can:")
print("1. Run the full BioMistral model (3.5GB download)")
print("2. Or use this working setup as a base")
print("\nTo load BioMistral, run:")
print("!python colab_app.py")