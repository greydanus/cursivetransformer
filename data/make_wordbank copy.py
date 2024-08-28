import numpy as np
import string
import argparse

def generate_word_bank(num_examples, min_length, max_length, uppercase_prob, capitalize_prob, period_prob, letter_probs, length_slope):
    letters = string.ascii_lowercase
    length_range = np.arange(min_length, max_length + 1)
    length_probs = 1 - length_slope * (length_range - min_length) / (max_length - min_length)
    length_probs = np.maximum(length_probs, 0)
    length_probs[0] *= .5
    length_probs[1] /= .8
    length_probs /= length_probs.sum()
    print('Length probs:', length_probs)
    
    word_bank = []
    for _ in range(num_examples):
        length = np.random.choice(length_range, p=length_probs)
        word = ''.join(np.random.choice(list(letters), size=length, p=letter_probs))
        word = word.upper() if np.random.random() < uppercase_prob else (word.capitalize() if np.random.random() < capitalize_prob else word)
        word += '.' if np.random.random() < period_prob else ''
        word_bank.append(word)
    return word_bank

def write_word_bank_to_file(word_bank, filename='synthbank.txt'):
    with open(filename, 'w') as f:
        f.write('const words = [' + ', '.join(f'"{w}"' for w in word_bank) + '];')

# python make_wordbank.py --num_examples 5000 --uppercase_prob 0 --capitalize_prob 0 --period_prob 0 --output 'synthbank.txt'
def main():
    parser = argparse.ArgumentParser(description='Generate a word bank')
    parser.add_argument('--num_examples', type=int, default=5000, help='Number of words to generate')
    parser.add_argument('--min_length', type=int, default=3, help='Minimum word length')
    parser.add_argument('--max_length', type=int, default=11, help='Maximum word length')
    parser.add_argument('--uppercase_prob', type=float, default=0.15, help='Probability of uppercase words')
    parser.add_argument('--capitalize_prob', type=float, default=0.25, help='Probability of capitalizing first letter')
    parser.add_argument('--period_prob', type=float, default=0.1, help='Probability of adding a period')
    parser.add_argument('--length_slope', type=float, default=0.85, help='Slope for length probability distribution')
    parser.add_argument('--output', default='capsbank.txt', help='Output filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)

    letter_probs = np.array([8.2, 1.5, 2.8, 4.3, 13, 2.2, 2, 6.1, 7, 0.15, 0.77, 4, 2.4, 6.7, 7.5, 1.9, 0.095, 6, 6.3, 9.1, 2.8, 0.98, 2.4, 0.15, 2, 0.074])
    letter_probs = np.maximum(letter_probs, 2a)
    letter_probs /= letter_probs.sum()

    word_bank = generate_word_bank(args.num_examples, args.min_length, args.max_length, args.uppercase_prob, args.capitalize_prob, args.period_prob, letter_probs, args.length_slope)
    write_word_bank_to_file(word_bank, args.output)

if __name__ == '__main__':
    main()