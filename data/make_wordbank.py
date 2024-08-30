import numpy as np
import string
import argparse
import json
from collections import Counter

def generate_word_bank(args):
    letters = string.ascii_lowercase
    length_range = np.arange(args.min_length, args.max_length + 1)
    length_probs = 1 - args.length_slope * (length_range - args.min_length) / (args.max_length - args.min_length)
    length_probs = np.maximum(length_probs, 0)
    length_probs[0] *= .5
    length_probs[1] /= .8
    length_probs /= length_probs.sum()
    # print('Length probs:', length_probs)
    
    word_bank = []
    for _ in range(args.num_examples):
        if args.add_digits and np.random.random() < args.digit_word_prob:
            length = np.random.randint(1, args.digit_max_length + 1)
            word = ''.join(np.random.choice(list('0123456789'), size=length))
            if np.random.random() < args.digit_decimal_prob:
                decimal_pos = np.random.randint(0, len(word) + 1)
                word = word[:decimal_pos] + '.' + word[decimal_pos:]
        else:
            length = np.random.choice(length_range, p=length_probs)
            word = ''.join(np.random.choice(list(letters), size=length, p=args.letter_probs))
            word = word.upper() if np.random.random() < args.uppercase_prob else (word.capitalize() if np.random.random() < args.capitalize_prob else word)
        
        # Add quote at the beginning, end, or middle of the word
        if np.random.random() < args.quote_prob:
            quote = np.random.choice(['"', "'"])
            if quote == "'" and np.random.random() < 0.5 and len(word) > 1:
                # Insert single quote in the middle for contractions
                insert_pos = np.random.randint(1, len(word))
                word = word[:insert_pos] + "'" + word[insert_pos:]
            elif np.random.random() < 0.5:
                word = quote + word
            else:
                word = word + quote
        
        # Add parenthesis at the beginning or end of the word
        if np.random.random() < args.parenthesis_prob:
            if np.random.random() < 0.5:
                word = '(' + word
            else:
                word = word + ')'
        
        # Add sentence flow punctuation at the end of the word
        if np.random.random() < args.sentence_flow_prob:
            for punctuation, prob in zip(args.punctuations, args.punctuation_probs):
                if np.random.random() < prob:
                    word = word + punctuation
                    break
        
        word_bank.append(word)
    return word_bank

def write_word_bank_to_file(word_bank, filename='synthbank.txt'):
    with open(filename, 'w') as f:
        f.write('const words = ' + json.dumps(word_bank) + ';')

def analyze_word_bank(word_bank, k=75):
    first_k_words = word_bank[:k]  # Print first k words
    
    print(f"First {k} words:")
    line_width = 0
    max_line_width = 80  # Adjust this value based on your terminal width
    for word in first_k_words:
        if line_width + len(word) + 1 > max_line_width:
            print()  # Start a new line
            line_width = 0
        print(word, end=" ")
        line_width += len(word) + 1
    print("\n")  # Print two newlines at the end

    all_chars = set(''.join(word_bank))
    char_counts = Counter(''.join(word_bank))
    total_chars = sum(char_counts.values())
    
    char_probs = {char: count / total_chars for char, count in char_counts.items()}
    
    # Sort characters by probability (descending) and format output
    sorted_probs = sorted(char_probs.items(), key=lambda x: x[1], reverse=True)
    max_char_width = max(len(repr(char)) for char, _ in sorted_probs)
    max_prob_width = max(len(f"{prob:.2%}") for _, prob in sorted_probs)
    
    print("Character probabilities:")
    line_width = 0
    max_line_width = 80  # Adjust this value based on your terminal width
    for char, prob in sorted_probs:
        formatted_prob = f"{repr(char):<{max_char_width}} : {prob:.2%}"
        if line_width + len(formatted_prob) + 2 > max_line_width:
            print()  # Start a new line
            line_width = 0
        print(formatted_prob, end="  ")
        line_width += len(formatted_prob) + 2
    print()  # Print a newline at the end

    print("\nCharacter counts:")
    line_width = 0
    max_count_width = max(len(str(count)) for count in char_counts.values())
    for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
        formatted_count = f"{repr(char):<{max_char_width}} : {count:>{max_count_width}}"
        if line_width + len(formatted_count) + 2 > max_line_width:
            print()  # Start a new line
            line_width = 0
        print(formatted_count, end="  ")
        line_width += len(formatted_count) + 2
    print()  # Print a newline at the end
    
    print("\nFull alphabet of all characters used:")
    print(''.join(sorted(all_chars, key=lambda x: -char_probs[x])))

# python make_wordbank.py --num_examples 5000 --uppercase_prob 0 --capitalize_prob 0 --period_prob 0 --output 'synthbank.txt'
def main():
    parser = argparse.ArgumentParser(description='Generate a word bank')
    parser.add_argument('--num_examples', type=int, default=2000, help='Number of words to generate')
    parser.add_argument('--min_length', type=int, default=3, help='Minimum word length')
    parser.add_argument('--max_length', type=int, default=10, help='Maximum word length')
    parser.add_argument('--uppercase_prob', type=float, default=0.25, help='Probability of uppercase words')
    parser.add_argument('--capitalize_prob', type=float, default=0.25, help='Probability of capitalizing first letter')
    parser.add_argument('--length_slope', type=float, default=0.85, help='Slope for length probability distribution')
    parser.add_argument('--output', default='bigbank.txt', help='Output filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')
    parser.add_argument('--add_digits', action='store_true', default=True, help='Add digit words to the word bank')
    parser.add_argument('--digit_max_length', type=int, default=5, help='Maximum length for digit words')
    parser.add_argument('--digit_decimal_prob', type=float, default=0.3, help='Probability of adding a decimal point to digit words')
    parser.add_argument('--digit_word_prob', type=float, default=0.25, help='Probability of generating a digit word')
    parser.add_argument('--quote_prob', type=float, default=0.1, help='Probability of adding a quote (single or double) to a word')
    parser.add_argument('--parenthesis_prob', type=float, default=0.1, help='Probability of adding a parenthesis (open or close) to a word')
    parser.add_argument('--sentence_flow_prob', type=float, default=0.80, help='Probability of adding sentence flow punctuation to a word')
    parser.add_argument('--punctuations', nargs='+', default=[",", ".", "?", "!"], help='List of punctuations to use')
    parser.add_argument('--punctuation_probs', nargs='+', type=float, default=[0.05, 0.05, 0.05, 0.05], help='Probabilities for each punctuation')
    args = parser.parse_args()

    np.random.seed(args.seed)

    letter_probs = np.array([6, 1.5, 2.8, 4.3, 6, 2.2, 2, 6.1, 6, 4, 0.77, 4, 2.4, 6, 6, 1.9, 0.095, 6, 6.3, 6, 2.8, 0.98, 2.4, 4, 2, 0.074])
    # letter_probs = np.array([6, 4, 4, 4.3, 6, 4, 4, 6.1, 6, 4, 4, 4, 4, 6, 6, 4, 4, 6, 6.3, 6, 4, 4, 4, 4, 4, 4])
    letter_probs = np.maximum(letter_probs, 4)
    args.letter_probs = letter_probs / letter_probs.sum()

    word_bank = generate_word_bank(args)
    analyze_word_bank(word_bank)
    write_word_bank_to_file(word_bank, args.output)

if __name__ == '__main__':
    main()