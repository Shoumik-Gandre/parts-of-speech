from collections import Counter
from itertools import chain
from pathlib import Path
from hmm.decoder.viterbi import ViterbiDecoderLog
from iodata import read_wordsequences, read_wordsequences_tagsequences
from metrics import accuracy_all
from vocabulary.vocab import Vocab
from preprocessing.count_replace import CountReplaceTransform
from preprocessing.vocab_encoder import VocabEncoderTransform
from hmm.model import HMM
from hmm.decoder.greedy import GreedyDecoder


def generate_vocab(input_path: str | Path, word_vocab_path: str | Path, tag_vocab_path: str | Path) -> None:
    word_sequences, tag_sequences = read_wordsequences_tagsequences(input_path)
    
    count_replacer = CountReplaceTransform(2, '<unk>').fit(sequences=word_sequences)
    new_seqs = count_replacer.transform(sequences=word_sequences)
    word_vocab = Vocab.from_sequences(new_seqs)
    word_vocab.to_file(word_vocab_path)

    tag_vocab = Vocab.from_sequences(tag_sequences)
    tag_vocab.to_file(tag_vocab_path)
    

def train_hmm(input_path: str | Path, word_vocab_path: str | Path, tag_vocab_path: str | Path, dev_path: str | Path) -> None:
    # Load Train
    word_sequences, tag_sequences = read_wordsequences_tagsequences(input_path)
    # Transform
    count_replacer = CountReplaceTransform(2, '<unk>').fit(word_sequences)
    word_sequences = count_replacer.transform(sequences=word_sequences)

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