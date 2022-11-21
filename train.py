import pandas as pd
from sklearn.model_selection import train_test_split

from hangmanai.model import train
from hangmanai.preprocess import create_input_sample, create_loader
from hangmanai.player import HangmanPlayer

from config import FILE_PATH, FRAC, RANDOM_STATE, MODEL_PATH, N_EPOCH

MODEL_NAME = 'bidiretionLSTM'
def main():
    df = pd.read_csv(FILE_PATH, header=None, na_filter=False)
    series_train, series_test = train_test_split(df[0], train_size=FRAC, random_state=RANDOM_STATE)
    features, label = create_input_sample(series_train)
    train_loader = create_loader(features, label)

    for epoch in range(N_EPOCH):
        model = train(train_loader, save_path=f'{MODEL_PATH}/{MODEL_NAME}_{epoch}.torch')
        model.eval()
        player = HangmanPlayer(model)
        _ = player.play(series_test)

if __name__ == '__main__':
    main()