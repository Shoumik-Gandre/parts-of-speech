from collections import Counter
from itertools import chain
from pathlib import Path
from hmm.decoder.viterbi import ViterbiDecoderLog
from iodata import read_wordsequences, read_wordsequences_tagsequences
from metrics import accuracy_all
from vocabulary.vocab import Vocab
from preprocessing.count_replace import CountReplaceTransform
from preprocessing.regex_replace import RegexCountReplaceTransform
from preprocessing.pseudowords.pseudoword_replace import PseudoWordReplaceTransform
from hmm.model import HMM
from hmm.decoder.greedy import GreedyDecoder

UNK_TOKEN = '< UNK >'


def generate_vocab(input_path: str | Path, word_vocab_path: str | Path, tag_vocab_path: str | Path) -> None:
    word_sequences, tag_sequences = read_wordsequences_tagsequences(input_path)
    
    count_replacer = PseudoWordReplaceTransform(2, UNK_TOKEN)
    new_seqs = count_replacer.fit_transform(sequences=word_sequences)

    # Create Word Vocab
    word_vocab = Vocab.from_sequences(new_seqs)
    word_vocab.to_file(word_vocab_path)

    # Create Tag Vocab
    tag_vocab = Vocab.from_sequences(tag_sequences)
    tag_vocab.to_file(tag_vocab_path)


def train_hmm(input_path: str | Path, word_vocab_path: str | Path, tag_vocab_path: str | Path, dev_path: str | Path) -> None:
    # Load Train
    word_sequences, tag_sequences = read_wordsequences_tagsequences(input_path)
    # Transform
    count_replacer = PseudoWordReplaceTransform(2, UNK_TOKEN)
    word_sequences = count_replacer.fit_transform(sequences=word_sequences)

    # Load Vocab
    word_vocab = Vocab.from_file(word_vocab_path)
    tag_vocab = Vocab.from_file(tag_vocab_path)

    # Train Model
    hmm = HMM(set(word_vocab.keys()), set(tag_vocab.keys()))
    hmm.train(X=word_sequences, Y=tag_sequences)

    # Load Dev
    X_dev, y_dev = read_wordsequences_tagsequences(dev_path)
    X_dev = count_replacer.transform(sequences=X_dev)
    
    # Decoding
    gdecoder = GreedyDecoder(hmm)
    y_hat = [gdecoder.decode(sentence) for sentence in X_dev]
    print(f"[Dev]\t[Greedy Decoding] accuracy = {accuracy_all(y_dev, y_hat): .4f} / 1")

    # Viterbi Decode
    vdecoder = ViterbiDecoderLog(hmm)
    y_hat = [vdecoder.decode(sentence) for sentence in X_dev]

    print(f"[Dev]\t[Viterbi Decoding] accuracy = {accuracy_all(y_dev, y_hat): .4f} / 1")


def predict_hmm(input_path: str | Path, output_path: str | Path) -> None:
    pass


def generate_vocab2(input_path: str | Path, word_vocab_path: str | Path, tag_vocab_path: str | Path) -> None:
    word_sequences, tag_sequences = read_wordsequences_tagsequences(input_path)
    
    two_digit_replacer = RegexCountReplaceTransform(2, r'^[0-9]{2}$', '<unk>')
    new_seqs = two_digit_replacer.fit_transform(sequences=word_sequences)

    # Create Word Vocab
    word_vocab = Vocab.from_sequences(new_seqs)
    word_vocab.to_file(word_vocab_path)

    # Create Tag Vocab
    tag_vocab = Vocab.from_sequences(tag_sequences)
    tag_vocab.to_file(tag_vocab_path)