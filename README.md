English-Japanese Translator using Transformers
This project implements an English-to-Japanese translator using Transformer models. The datasets used are the Japanese-English bilingual corpus and the English-Japanese parallel corpus.

Setup
To run this project, you will need to install the following dependencies:


!pip install wget tensorflow kaggle
Data Acquisition
Upload Kaggle API Key:
Ensure you have a kaggle.json file containing your Kaggle API key. Upload it using the following command:


from google.colab import files
files.upload()
Setup Kaggle Configuration:


!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
Download Datasets:


!kaggle datasets download -d team-ai/japaneseenglish-bilingual-corpus
!kaggle datasets download -d nexdatafrank/english-japanese-parallel-corpus-data
Extract Datasets:


import zipfile

# Unzip the first dataset
with zipfile.ZipFile('japaneseenglish-bilingual-corpus.zip', 'r') as zip_ref:
    zip_ref.extractall('japaneseenglish-bilingual-corpus')

# Unzip the second dataset
with zipfile.ZipFile('english-japanese-parallel-corpus-data.zip', 'r') as zip_ref:
    zip_ref.extractall('english-japanese-parallel-corpus')
Data Inspection
List Files in the Dataset Directories:


import os

# List files in the first dataset directory
print("Files in 'japaneseenglish-bilingual-corpus':")
print(os.listdir('japaneseenglish-bilingual-corpus'))

# List files in the second dataset directory
print("Files in 'english-japanese-parallel-corpus':")
print(os.listdir('english-japanese-parallel-corpus'))
Load Specific Files:


import pandas as pd

# Load kyoto_lexicon.csv, skipping bad lines
kyoto_lexicon_path = 'japaneseenglish-bilingual-corpus/kyoto_lexicon.csv'
kyoto_lexicon = pd.read_csv(kyoto_lexicon_path, on_bad_lines='skip')
print("First few rows of kyoto_lexicon.csv:")
print(kyoto_lexicon.head())

# Extract wiki_corpus_2.01.tar
import tarfile

tar_path = 'japaneseenglish-bilingual-corpus/wiki_corpus_2.01.tar'
tar = tarfile.open(tar_path)
tar.extractall(path='japaneseenglish-bilingual-corpus/wiki_corpus_2.01')
tar.close()

# Load Wiki_Corpus_List_2.01.csv
wiki_corpus_path = 'japaneseenglish-bilingual-corpus/Wiki_Corpus_List_2.01.csv'
wiki_corpus = pd.read_csv(wiki_corpus_path)
print("First few rows of Wiki_Corpus_List_2.01.csv:")
print(wiki_corpus.head())
Preprocessing
Preprocess and Tokenize Sentences:


import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define a preprocessing function
def preprocess_sentence(sentence):
    return sentence.lower()

# Preprocess and tokenize the sentences
sentences['EN'] = sentences['EN'].apply(preprocess_sentence)
sentences['JA'] = sentences['JA'].apply(lambda x: preprocess_sentence(str(x)))

# Tokenize the English sentences
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(sentences['EN'])
eng_sequences = eng_tokenizer.texts_to_sequences(sentences['EN'])

# Tokenize the Japanese sentences
jpn_tokenizer = Tokenizer()
jpn_tokenizer.fit_on_texts(sentences['JA'])
jpn_sequences = jpn_tokenizer.texts_to_sequences(sentences['JA'])

# Pad the sequences
max_len = 100  # Adjust as per your requirement
eng_padded = pad_sequences(eng_sequences, padding='post', maxlen=max_len)
jpn_padded = pad_sequences(jpn_sequences, padding='post', maxlen=max_len)
Model Definition and Training
Define the Transformer Model:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, Add

# Define Transformer Encoder Layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# Define Transformer Decoder Layer
def transformer_decoder(inputs, encoder_outputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, encoder_outputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# Define model inputs
encoder_inputs = Input(shape=(max_len,))
decoder_inputs = Input(shape=(max_len-1,))

# Embedding layers
encoder_embedding = Embedding(input_dim=len(eng_tokenizer.word_index) + 1, output_dim=256)(encoder_inputs)
decoder_embedding = Embedding(input_dim=len(jpn_tokenizer.word_index) + 1, output_dim=256)(decoder_inputs)

# Encoder
encoder_outputs = transformer_encoder(encoder_embedding, head_size=256, num_heads=4, ff_dim=256, dropout=0.1)

# Decoder
decoder_outputs = transformer_decoder(decoder_embedding, encoder_outputs, head_size=256, num_heads=4, ff_dim=256, dropout=0.1)

# Final dense layer
decoder_dense = Dense(len(jpn_tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Display the model summary
model.summary()
Train the Model:


history = model.fit(
    [eng_train, jpn_train[:, :-1]],  # Input: English sequences and Japanese input sequences
    np.expand_dims(jpn_train[:, 1:], -1),  # Output: Japanese target sequences
    batch_size=64,
    epochs=10,
    validation_data=([eng_val, jpn_val[:, :-1]], np.expand_dims(jpn_val[:, 1:], -1))
)
Visualize Training History:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
Conclusion
This project demonstrates how to build an English-to-Japanese translator using Transformer models. The model is trained on sentence pairs from bilingual corpora and can be further refined with more data and hyperparameter tuning.
