# parts-of-speech
Use a Bigram HMM to tag the parts of speech in a sentence

## Commands
Use the following commands when inside src

### Generate Vocabulary:
python main.py -a vocab -i path/to/input/data/ -w path/to/word/vocab/file -t path/to/tag/vocab/file

### Train HMM:
python main.py -a train -i path/to/train/file/ -d path/to/dev/file -w path/to/word/vocab/file -t path/to/tag/vocab/file

