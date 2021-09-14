from collections import Counter
import pickle


class Vocab(object):
    def __init__(self, logs, specials=['PAD', 'UNK']):
        self.pad_index = 0
        self.unk_index = 1

        self.stoi = {}
        self.itos = list(specials)

        event_count = Counter()
        for line in logs:
            for logkey in line.split():
                event_count[logkey] += 1

        for event, freq in event_count.items():
            self.itos.append(event)

        self.stoi = {e: i for i, e in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def save_vocab(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_vocab(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

