"""
Gradio web interface for Luganda-English translation
"""

import os
import gradio as gr
import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

# Configuration
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
CUSTOM_MODEL = os.environ.get("HF_MODEL_ID", None)

print("Initializing Luganda-English translator...")

# Load models and tokenizer
try:
    # Always load the base NLLB tokenizer to ensure we have all language codes
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    print("Loaded tokenizer from base NLLB model")
    
    # Try to load the custom fine-tuned model if specified
    if CUSTOM_MODEL:
        print(f"Attempting to load custom model: {CUSTOM_MODEL}")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                CUSTOM_MODEL,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print(f"Successfully loaded custom model: {CUSTOM_MODEL}")
            model_name = CUSTOM_MODEL
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            print("Falling back to base NLLB model")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                DEFAULT_MODEL,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None  # Add this line
            )
            model_name = DEFAULT_MODEL
    else:
        # Use the default model
        print(f"Loading default model: {DEFAULT_MODEL}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            DEFAULT_MODEL,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None  # Add this line
        )
        model_name = DEFAULT_MODEL
    
    print(f"Model loaded: {model_name}")
    
    if torch.cuda.is_available():
        print("CUDA is available - using GPU")
    else:
        print("CUDA is not available - using CPU")
        
except OSError as e:
    print(f"Network or file error: {e}")
    raise
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("GPU memory error - falling back to CPU")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            DEFAULT_MODEL,
            device_map=None,
            low_cpu_mem_usage=True
        )
        model_name = DEFAULT_MODEL + " (CPU fallback)"
    else:
        print(f"Runtime error: {e}")
        raise
except Exception as e:
    print(f"Unexpected error loading models: {e}")
    raise

def translate_lug_to_eng(text):
    """Translate from Luganda to English"""
    if not text or not text.strip():
        return "Please enter text to translate."
    
    try:
        tokenizer.src_lang = "lug_Latn"
        inputs = tokenizer(text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        return f"Translation error: {str(e)}"

def translate_eng_to_lug(text):
    """Translate from English to Luganda"""
    if not text or not text.strip():
        return "Please enter text to translate."
    
    try:
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["lug_Latn"],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        return f"Translation error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Luganda-English Translator") as demo:
    gr.Markdown("# Luganda-English Translator")
    gr.Markdown("Translate between Luganda and English using the NLLB model.")
    
    with gr.Tab("Luganda to English"):
        with gr.Row():
            with gr.Column():
                lug_input = gr.Textbox(label="Luganda Text", lines=5, placeholder="Enter text in Luganda...")
                lug_to_eng_btn = gr.Button("Translate to English")
            with gr.Column():
                eng_output = gr.Textbox(label="English Translation", lines=5)
        
        # Add examples for Luganda to English
        gr.Examples(
            examples=[
                ["Katonda"],
                ["Omukazi omuŋŋanzi"],
                ["Baibuli Entukuvu"],
                ["Okwagala kwa Katonda"]
            ],
            inputs=lug_input,
            outputs=eng_output,
            fn=translate_lug_to_eng,
            cache_examples=True
        )
        
        lug_to_eng_btn.click(translate_lug_to_eng, inputs=lug_input, outputs=eng_output)
        
    with gr.Tab("English to Luganda"):
        with gr.Row():
            with gr.Column():
                eng_input = gr.Textbox(label="English Text", lines=5, placeholder="Enter text in English...")
                eng_to_lug_btn = gr.Button("Translate to Luganda", variant="primary")
            with gr.Column():
                lug_output = gr.Textbox(label="Luganda Translation", lines=5)
        
        # Add examples for English to Luganda
        gr.Examples(
            examples=[
                ["God"],
                ["The brave woman"],
                ["Holy Bible"],
                ["The love of God"]
            ],
            inputs=eng_input,
            outputs=lug_output,
            fn=translate_eng_to_lug,
            cache_examples=True
        )
        
        eng_to_lug_btn.click(translate_eng_to_lug, inputs=eng_input, outputs=lug_output)

    gr.Markdown("### Model Information")
    gr.Markdown(f"Using model: **{model_name}**")
    
    gr.Markdown("""
    ### About
    
    This translator uses Meta's NLLB (No Language Left Behind) model, fine-tuned on Luganda-English data.
    
    Source code: [GitHub Repository](https://github.com/kserumaga/luganda-translator)
    """)

# Launch the app with proper settings for Hugging Face Spaces
if __name__ == "__main__":
    demo.queue()  # Enable queuing for better handling of multiple users
    demo.launch(
        share=False,  # Don't create a public link
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860  # Default port for HF Spaces
    )