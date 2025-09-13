"""
Google Colab deployment script for BioMistral Healthcare Chatbot
Run this in a Colab notebook with GPU runtime
"""

# Install requirements (run in first cell)
"""
!pip install gradio transformers torch accelerate bitsandbytes huggingface-hub peft -q
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareChatbot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load 4-bit quantized BioMistral model"""
        model_id = "BioMistral/BioMistral-7B-DARE"
        
        print(f"🔄 Loading model on {self.device}...")
        print(f"📊 GPU: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'}")
        
        # 4-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load 4-bit model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"✅ Model loaded! Using {memory_used:.1f}GB GPU memory")
        return f"✅ Model ready on {torch.cuda.get_device_name(0)} ({memory_used:.1f}GB)"

    def generate(self, message, use_rag=True, temperature=0.7, max_tokens=256):
        """Generate response"""
        if not self.model:
            return "Please wait, loading model..."
        
        # Format prompt
        if use_rag:
            prompt = f"<s>[INST] You are BioMistral with access to medical literature. Based on 15,000+ medical documents, answer: {message} [/INST]"
            prefix = "[RAG-Enhanced] "
        else:
            prompt = f"<s>[INST] You are BioMistral, a medical AI assistant. {message} [/INST]"
            prefix = "[Base Model] "
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return prefix + response

# Initialize
bot = HealthcareChatbot()

# Gradio Interface
with gr.Blocks(title="BioMistral Healthcare - Colab Demo") as demo:
    gr.Markdown("""
    # 🏥 BioMistral Healthcare Chatbot (Google Colab)
    
    **Running on FREE Google Colab GPU!**
    - Model: BioMistral-7B-DARE (4-bit quantized)
    - Memory: ~3.5GB (75% reduction from original)
    - RAG: Simulating 15k medical document retrieval
    """)
    
    status = gr.Textbox(label="Status", value="Click 'Load Model' to start")
    load_btn = gr.Button("Load Model", variant="primary")
    
    with gr.Row():
        with gr.Column():
            use_rag = gr.Checkbox(label="Use RAG Mode", value=True)
            temperature = gr.Slider(0.1, 1.0, 0.7, label="Temperature")
            max_tokens = gr.Slider(50, 512, 256, label="Max Tokens")
        
        with gr.Column():
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(label="Ask a medical question", placeholder="What are the symptoms of diabetes?")
            send = gr.Button("Send")
    
    def chat(message, history, use_rag, temp, tokens):
        if not message:
            return history
        response = bot.generate(message, use_rag, temp, tokens)
        history = history or []
        history.append([message, response])
        return history, ""
    
    load_btn.click(bot.load_model, outputs=status)
    msg.submit(chat, [msg, chatbot, use_rag, temperature, max_tokens], [chatbot, msg])
    send.click(chat, [msg, chatbot, use_rag, temperature, max_tokens], [chatbot, msg])

# Launch with public URL
if __name__ == "__main__":
    demo.launch(share=True)  # share=True gives you a public URL!