import pandas as pd
from sklearn.model_selection import train_test_split

from hangmanai.player import HangmanPlayer
from hangmanai.model import load_model

from config import FILE_PATH, FRAC, RANDOM_STATE, MODEL_PATH

MODEL_NAME = 'bidiretionLSTM_0'
def test():
    df = pd.read_csv(FILE_PATH, header=None, na_filter=False)
    _, series_test = train_test_split(df[0], train_size=FRAC, random_state=RANDOM_STATE)
    model = load_model(f'{MODEL_PATH}/{MODEL_NAME}.torch')
    model.eval()
    player = HangmanPlayer(model)
    success_rate = player.play(series_test, verbose=True)

    return success_rate

if __name__ == '__main__':
    success_rate = test()