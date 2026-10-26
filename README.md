# Next Word Predictor using LSTM (PyTorch)

A next-word prediction model built from scratch with **PyTorch** and an **LSTM** network. Given a few starting words, the model predicts the next word and can keep going to generate a whole phrase, one word at a time.

The model is trained on a small FAQ-style text corpus (a course FAQ document), which makes it a good hands-on project for understanding how sequence models and text preprocessing work end to end.

---

## Features

- Word-level tokenization with NLTK
- Custom vocabulary with an `<unk>` token for unseen words
- Training sequences built with the n-gram prefix approach
- Left-padding to a fixed length
- Custom PyTorch `Dataset` and `DataLoader`
- `Embedding -> LSTM -> Linear` architecture
- Auto-regressive text generation (predict a word, append it, repeat)
- Accuracy evaluation function

---

## How It Works

1. **Tokenize**: the document is lowercased and split into word tokens using `nltk.word_tokenize`.
2. **Build vocabulary**: each unique token gets an integer ID. ID `0` is reserved for `<unk>` and is also used as the padding value.
3. **Create training sequences**: every sentence (line) is expanded into growing prefixes. For example, `[a, b, c, d]` becomes `[a, b]`, `[a, b, c]`, `[a, b, c, d]`.
4. **Pad**: all sequences are left-padded with zeros to the maximum length (62).
5. **Split into X and y**: the first 61 tokens are the input and the last token is the target word.
6. **Train**: the LSTM learns to predict the target word with cross-entropy loss.
7. **Generate**: predict the next word, append it to the input, and repeat.

---

## Model Architecture

```
Input (batch, 61)
   |
Embedding (vocab_size -> 100)
   |
LSTM (input 100, hidden 150, batch_first)
   |
Linear (150 -> vocab_size)
   |
Output logits over the vocabulary
```

| Hyperparameter | Value |
| --- | --- |
| Embedding dimension | 100 |
| LSTM hidden size | 150 |
| Batch size | 32 |
| Epochs | 50 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

---

## Results

| Metric | Value |
| --- | --- |
| Loss (epoch 1) | 165.11 (summed over batches) |
| Loss (epoch 50) | 4.34 (summed over batches) |
| Accuracy | 80.65% |

> **Note:** accuracy is measured on the same data the model was trained on. The dataset is tiny, so the model largely memorizes the text. This project is meant for learning, not for production use.

### Sample generation

Input: `The course follows`

```
The course follows a
The course follows a monthly
The course follows a monthly subscription
The course follows a monthly subscription model
The course follows a monthly subscription model where
The course follows a monthly subscription model where you
The course follows a monthly subscription model where you have
The course follows a monthly subscription model where you have to
The course follows a monthly subscription model where you have to make
The course follows a monthly subscription model where you have to make monthly
```

---

## Tech Stack

- Python 3
- PyTorch
- NLTK
- NumPy
- Jupyter Notebook / Google Colab

---

## Usage

```python
prediction(model, vocab, "The course follows a monthly")
# -> "The course follows a monthly subscription"
```

To generate several words in a row:

```python
input_text = "The course follows"
for _ in range(10):
    input_text = prediction(model, vocab, input_text)
    print(input_text)
```

---

## Project Structure

```
.
├── Nxt_word_predictor_LSTM.ipynb   # Full code: preprocessing, training, prediction
└── README.md
```

---

## Author

**Gulshan**
Electronics and Communication Engineering, MNIT Jaipur
