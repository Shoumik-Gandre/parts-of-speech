from argparse import ArgumentParser, Namespace
from pathlib import Path
from perform import generate_vocab, train_hmm


def main(args: Namespace):
    match args.action:
        case 'vocab':
            generate_vocab(args.input_path, args.word_vocab_path, args.tag_vocab_path)

        case 'train':
            input_path = args.input_path
            dev_path = args.dev_path
            word_vocab_path = args.word_vocab_path
            tag_vocab_path = args.tag_vocab_path
            train_hmm(input_path, word_vocab_path, tag_vocab_path, dev_path)

        case 'predict':
            pass
    
    print("[DONE]")


if __name__  == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--action', '-a', choices=['vocab', 'train', 'predict'], type=str)
    parser.add_argument('--input-path', '-i', type=str)
    parser.add_argument('--dev-path', '-d', type=str)
    parser.add_argument('--word-vocab-path', '-w', type=str)
    parser.add_argument('--tag-vocab-path', '-t', type=str)
    args = parser.parse_args()
    main(args)