# Solving [Hangman](https://en.wikipedia.org/wiki/Hangman_(game)) Game using Neural Network
![alt text](https://github.com/aarontong95/HangmanAI/blob/main/doc/hangman.png)
<br> The solution is inspired by the training approach of [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) which is MLM(Masked Language Model). It is a fill-in-the-blank task, where a model uses the context words surrounding a mask token to try to predict what the masked word should be. For example:
```
Input Text: Hangman game is so [MASK]
Label: [MASK] = interesting
```
In the game [Hangman](https://en.wikipedia.org/wiki/Hangman_(game)), it would be:
```
Input Text: HANG#AN
Label: # = M`
```
## ENVIRONMENT
* python3.6

## Clone the Project
<pre>
git clone https://github.com/aarontong95/HangmanAI.git
</pre>

## Training
64 GB ram is need for training. Otherwise you can lower the value of FRAC in config.py which is the proportion of the train split
<pre>
python train.py
</pre>

## Testing
The successful rate is about 44% in the out of sample testing
<pre>
python test.py
</pre>

## What have implemented 
* Generate training sample for the model to learn (preprocess.py)
* Bidirectional LSTM as the model (model.py)
* Hangman Game Enviorment (game.py)
* Play Hangman Game with the model (palyer.py)
* More details in Solution.ipynb
