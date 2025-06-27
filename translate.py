"""
Translation utility for Luganda-English using NLLB model
"""

import argparse
import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from the given path or HF Hub ID."""
    print(f"Loading model from {model_path}")
    
    # Always use the base tokenizer to ensure we have all language codes
    try:
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        print("Loaded tokenizer from base NLLB model")
    except Exception as e:
        print(f"Failed to load base tokenizer: {e}")
        # Try to load from the specified model path
        tokenizer = NllbTokenizer.from_pretrained(model_path)
        print(f"Loaded tokenizer from {model_path}")
    
    # Load the model
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        print("Falling back to base NLLB model")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    return model, tokenizer

def translate(text, model, tokenizer, source_lang, target_lang):
    """Translate text between languages using the loaded model."""
    # Set source language
    tokenizer.src_lang = source_lang
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate translation with beam search
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    # Decode the output tokens
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translation

def main():
    parser = argparse.ArgumentParser(description="Translate between Luganda and English")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M", 
                        help="Path or HF Hub ID to the model")
    parser.add_argument("--direction", type=str, choices=["lug-eng", "eng-lug"], default="lug-eng",
                        help="Translation direction")
    parser.add_argument("--file", type=str, default=None, 
                        help="Optional: Path to a file containing text to translate (one sentence per line)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: Output file path for translated text")
    
    args = parser.parse_args()
    
    # Determine source and target languages
    if args.direction == "lug-eng":
        source_lang = "lug_Latn"
        target_lang = "eng_Latn"
    else:
        source_lang = "eng_Latn"
        target_lang = "lug_Latn"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Process file if provided
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        
        translations = []
        for i, line in enumerate(lines):
            print(f"Translating line {i+1}/{len(lines)}")
            translation = translate(line, model, tokenizer, source_lang, target_lang)
            translations.append(translation)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for translation in translations:
                    f.write(translation + "\n")
        else:
            for i, (original, translation) in enumerate(zip(lines, translations)):
                print(f"Line {i+1}:")
                print(f"  Original: {original}")
                print(f"  Translation: {translation}")
                print()
    
    # Process direct text input
    elif args.text:
        translation = translate(args.text, model, tokenizer, source_lang, target_lang)
        print(f"Original ({args.direction.split('-')[0]}): {args.text}")
        print(f"Translation ({args.direction.split('-')[1]}): {translation}")
    
    else:
        parser.error("Either --text or --file must be provided")

if __name__ == "__main__":
    main()