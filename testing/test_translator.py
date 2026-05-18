import sys
from pathlib import Path

# Setup paths so we can import modules from training
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"
TRANSLATIONS_DIR = TRAINING_DIR / "translations"

if str(TRANSLATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSLATIONS_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(1, str(TRAINING_DIR))

import lora_4
import lora_3

def test_translation():
    print("Initializing components...")
    # Load translator model and tokenizers
    reverse_model, reverse_inp_tokenizer, reverse_tar_tokenizer = lora_4.load_reverse_translator()
    
    # A single line of dialogue from Hamlet
    test_speeches = [
        "To be, or not to be, that is the question:",
        "O, that this too too solid flesh would melt, Thaw and resolve itself into a dew!"
    ]
    
    print("\nTranslating...")
    # Translate
    translated_speeches = lora_4.translate_speeches_with_reverse_model(
        test_speeches,
        reverse_model,
        reverse_inp_tokenizer,
        reverse_tar_tokenizer
    )
    
    # Print results
    for i, (orig, trans) in enumerate(zip(test_speeches, translated_speeches)):
        print(f"\nExample {i+1}:")
        print(f"Original: {orig}")
        print(f"Translated: {trans}")

if __name__ == "__main__":
    test_translation()
