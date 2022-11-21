import torch

from hangmanai.preprocess import preprocess_feature, output_to_char
from hangmanai.config import MAX_LEN, COVER_VALUE
from hangmanai.game import HangmanGame, State

class HangmanPlayer():
    def __init__(self, model):
        self.model = model
        self.word = ""
    def start_new_game(self, word):
        self.update_word(word)
        self.guessed = set()

    def update_word(self, word):
        self.gen = self.create_generator(word)

    def guess(self):
        word = next(self.gen)
        while word in self.guessed:
            word = next(self.gen)

        self.guessed.add(word)
        return word

    def create_generator(self, word):
        feature = preprocess_feature(word)
        featuresTest = torch.tensor(feature).to(torch.int64)
        featuresTest = featuresTest.reshape(1, MAX_LEN)
        proba = self.model(featuresTest)
        proba = torch.nn.functional.softmax(proba[featuresTest==COVER_VALUE], dim=1).sum(dim=0)
        _, indices = torch.sort(proba, descending=True)
        for i in indices:
            yield output_to_char(i.item())

    def play(self, series_test, verbose=False):
        n_games = len(series_test)
        n_success = 0
        success = []
        for word in series_test:
            game = HangmanGame(word, verbose=verbose)
            self.start_new_game(game.word_covered)
            is_run = True
            while is_run:
                guess_letter = self.guess()
                status = game.input(guess_letter)
                if status == State.FAIL:
                    is_run = False
                elif status == State.SUCCESS:
                    is_run = False
                    n_success += 1
                    success.append(word)
                elif status == State.RIGHT:
                    self.update_word(game.word_covered)

        success_rate = n_success/n_games*100
        print(f'Successful Rate: {success_rate}')
        return success_rate