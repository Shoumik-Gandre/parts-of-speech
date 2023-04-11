# parts-of-speech
Use a Bigram HMM to tag the parts of speech in a sentence

## About Preprocessing
There are two methods to handle unknown words:
1. Replace them with an Unknown symbol
2. Replace them with a pseudoword symbol

## Commands
Use the following commands when inside src

### Generate Vocabulary:
python main.py -a vocab -i path/to/input/data/ -w path/to/word/vocab/file -t path/to/tag/vocab/file

### Train HMM:
python main.py -a train -i path/to/train/file/ -d path/to/dev/file -w path/to/word/vocab/file -t path/to/tag/vocab/file

