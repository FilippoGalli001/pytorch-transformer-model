import torch
import torch.nn as nn
import math

# Convert sentence in vector of size 512
# Input Embedding

# Define class InputEmbeddings that extends nn.Module
class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # Memorizza la dimensione del modello (lunghezza del vettore di embedding per parola)
        self.d_model = d_model
        # Memorizza la dimensione del vocabolario (numero totale di parole uniche nel vocabolario)
        self.vocab_size =vocab_size
        # Crea un layer di embedding che trasforma indici di parole in vettori densi
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Calcola l'embedding per ogni indice di parola nell'input x
        # x è una sequenza di indici di parole. self.embedding(x) restituisce un tensore (batch, seq_len, d_model)
        # Scala gli embedding moltiplicandoli per la radice quadrata di d_model per mantenere la stabilità numerica
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# Position Embedding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # Max sequence length
        self.dropout = nn.Dropout(dropout) # Used to avoid overfitting

        # Build a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Crea un vettore con valori da 0 a seq_len - 1, ognuno rappresenta una posizione nella sequenza
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float()* (-math.log(10000.0) / d_model))
        # Apply the sin to even positions (from zero up to the end going forward by two)
        pe[:, 0::2] = torch.sin(position*div_term)
        # Apply the cos to odd positions (from one up to the end going forward by two)
        pe[:, 1::2] = torch.cos(position*div_term)

        # Aggiunge una dimensione di batch al tensore pe, quindi diventa (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Registra pe come buffer del modulo. ciò significa che pe non è un parametro addestrabile ma verrà salvato e caricato con il modello.
        self.register_buffer('pe', pe)

    def forward(self,x):
        # Add the positional encoding to every word inside the sentence
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization:
    
    # Use epsilon for numerical stability (avoid big and small numbers)
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # Multiplied
        self.alpha = nn.Parameter(torch.ones(1)) ##nn.Parameter makes the parameter learnable
        # Added
        self.bias = nn.Parameter(torch.zeros(1)) ##nn.Parameter makes the parameter learnable

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # Crea un layer lineare che trasforma un vettore di dimensione d_model in un vettore di dimensione d_ff
        # Questo layer applica una matrice di pesi W1 e un bias B1
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout) 
        # Crea un secondo layer lineare che trasforma un vettore di dimensione d_ff in un vettore di dimensione d_model
        # Questo layer applica una matrice di pesi W2 e un bias B2
        self.linear_2 = nn.Linear(d_model, d_ff) # W2 and B2


    # Il metodo forward applica una sequenza di trasformazioni al tensore x. 
    # Prima, linear_1 espande la dimensione del vettore e la funzione di attivazione ReLU viene applicata per introdurre non linearità.
    # Successivamente, il dropout viene applicato per disattivare casualmente alcune unità, e infine linear_2 riduce la dimensione del vettore per riportarlo a d_model.
    # Questo processo consente al modello di apprendere rappresentazioni più complesse e non lineari.
    def forward(self, x):
        # Applica la trasformazione lineare con la funzione di attivazione ReLU
        # Input: (batch, seq_len, d_model) -> Output: (batch, seq_len, d_ff)
        x = torch.relu(self.linear_1(x))
        # Applica il dropout
        x = self.dropout(x)
        # Applica la seconda trasformazione lineare
        # Output: (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(x)