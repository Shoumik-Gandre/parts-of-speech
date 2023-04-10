from pathlib import Path
import re
import json


def read_wordsequences_tagsequences(path: str | Path) -> tuple[list[list[str]], list[list[str]]]:
    """This is used to read from the train and dev set"""
    with open(path) as f:
        data = f.read()
    tagged_sentences = re.split(r'\n\n', data.strip('\n'))

    word_sequences = [
        [
            wordrecord.split('\t')[1] 
            for wordrecord in sentence.split('\n')
        ] 
        for sentence in tagged_sentences
    ]

    tag_sequences = [
        [
            tag_word.split('\t')[2] 
            for tag_word in sentence.split('\n')
        ] 
        for sentence in tagged_sentences
    ]
    
    return word_sequences, tag_sequences


def read_wordsequences(path: str | Path) -> list[list[str]]:
    """This is used to read from the train and dev set"""
    with open(path) as f:
        data = f.read()
    tagged_sentences = re.split(r'\n\n', data.strip('\n'))

    word_sequences = [
        [
            wordrecord.split('\t')[1] 
            for wordrecord in sentence.split('\n')
        ] 
        for sentence in tagged_sentences
    ]
    
    return word_sequences


def write_vocabulary(path: str, frequency: list[tuple[str, int]]) -> None:

    result = ""
    for index, (word, count) in enumerate(frequency):
        result += f"{word}\t{index}\t{count}\n"

    with open(path, 'w') as f:
        f.write(result)



def write_wordsequences_tagsequences(path: str, wordsequences: list[list[str]], tagsequences: list[list[str]]):
    
    result = ""
    for wordseq, tagseq in zip(wordsequences, tagsequences):
        for i, (word, tag) in enumerate(zip(wordseq, tagseq)):
            result += f"{i+1}\t{word}\t{tag}\n"
        result += "\n"
    result = result[:-1]
    
    with open(path, 'w') as f:
        f.write(result)
