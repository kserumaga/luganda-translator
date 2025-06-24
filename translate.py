import argparse
import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

def translate(text, model_path, source_lang="lug_Latn", target_lang="eng_Latn"):
    """
    Translate text using the fine-tuned model.
    
    Args:
        text: Text to translate
        model_path: Path to the saved model
        source_lang: Source language code (default: Luganda)
        target_lang: Target language code (default: English)
    
    Returns:
        Translated text
    """
    # Load the tokenizer and model
    print(f"Loading model from {model_path}...")
    tokenizer = NllbTokenizer.from_pretrained(model_path, src_lang=source_lang, tgt_lang=target_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate translation
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    # Decode the translation
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation

def main():
    parser = argparse.ArgumentParser(description="Translate text using the fine-tuned model")
    parser.add_argument("--input", type=str, required=True, help="Text to translate or path to a file containing text")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--direction", type=str, choices=["lug-eng", "eng-lug"], default="lug-eng", help="Translation direction")
    parser.add_argument("--output", type=str, help="Optional path to save translation output to a file")
    
    args = parser.parse_args()
    
    # Set source and target languages based on direction
    if args.direction == "lug-eng":
        source_lang = "lug_Latn"
        target_lang = "eng_Latn"
    else:
        source_lang = "eng_Latn" 
        target_lang = "lug_Latn"
    
    # Check if input is a file or text
    if args.input.endswith('.txt'):
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File {args.input} not found.")
            return
    else:
        text = args.input
    
    # Translate the text
    translation = translate(text, args.model_path, source_lang, target_lang)
    
    # Print the results
    print("\nOriginal text:")
    print(text)
    print("\nTranslation:")
    print(translation)
    
    # Save to output file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(translation)
        print(f"\nTranslation saved to {args.output}")

if __name__ == "__main__":
    main()