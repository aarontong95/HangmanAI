from hangmanai.config import ALLOW_GUESS, COVER_SYMBOL

class State():
    SUCCESS = 'SUCCESS'
    FAIL = 'FAIL'
    RIGHT = 'RIGHT'
    WRONG = 'WRONG'

class HangmanGame():

    def __init__(self, word, verbose):
        self.word = word
        self.tries_remains = ALLOW_GUESS
        self.need_guess = set()
        self.word_covered = ""
        self.n_word = len(word)
        self.verbose = verbose
        self.start()

    def start(self):
        self.word_covered = [COVER_SYMBOL] * self.n_word
        for letter in set(self.word):
            self.need_guess.add(letter)

        if self.verbose:
            print(f'Successfully start a new game! Real Word : {self.word}')


    def _print_status(func):
        def helper(*args, **kwargs):
            status = func(*args, **kwargs)
            self = args[0]
            if self.verbose:
                word_str = ''.join(self.word_covered)
                print(f'Guess Letter: {args[1]}, Status : {status}, Tries Remains {self.tries_remains}, Covered Word: {word_str}')
            return status
        return helper

    @_print_status
    def input(self, letter):
        if letter not in self.need_guess:
            self.tries_remains -= 1
            if self.tries_remains <= 0:
                return State.FAIL
            else:
                return State.WRONG

        else:
            self.need_guess.remove(letter)
            word_covered_new = self.word_covered.copy()
            for i in range(self.n_word):
                if self.word[i] == letter:
                    self._fill_word(i, word_covered_new, letter)

            self.word_covered = word_covered_new
            if not len(self.need_guess):
                return State.SUCCESS
            else:
                return State.RIGHT

    def _fill_word(self, i, word_covered, letter):
        if letter == self.word[i]:
            word_covered[i] = letter
