# BERT

BERT (Bidirectional Encoder Representations from Transformers) is a deep learning model designed to understand the context of words in a sentence by considering both the left and right context simultaneously.


BERT's input is a sequence of tokens. Each token is converted into a vector representation by summing three embeddings:
1) Token Embedding: Represents the meaning of the word.
2) Segment Embedding: Indicates which sentence a token belongs to (useful for tasks involving sentence pairs).
3) Position Embedding: Encodes the position of the token in the sequence.
The input to BERT is a sequence of tokens represented as:
X = [x_1, x_2, ..., x_n]
where x_i is the vector for the i-th token in the sequence.

Since BERT uses a Transformer architecture without recurrence, positional encoding is added to the token embeddings to give the model information about the position of tokens in the sequence.

The positional encoding for the i-th token at dimension k is given by:
PE(i, 2k) = sin(i / 10000^(2k/d_model))
PE(i, 2k+1) = cos(i / 10000^(2k/d_model))
Here:
i is the position of the token in the sequence.
k is the dimension of the embedding.
d_model is the dimension of the token embeddings.

The core component of BERT is the self-attention mechanism. Self-attention allows each token to focus on different parts of the input sequence, providing a context-aware representation.

Given an input sequence, three matrices are derived:
1) Query (Q):
Q = X * W_Q
2) Key (K):
K = X * W_K
3) Value (V):
V = X * W_V
Here:
X is the input sequence matrix.
W_Q, W_K, and W_V are learned weight matrices.

The attention scores are computed using the dot product of the query and key matrices, scaled by the square root of the dimension of the key vectors:
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
Where:
d_k is the dimension of the key vectors.
softmax is the softmax function that normalizes the attention scores.


BERT uses multi-head attention, where multiple sets of Q, K, and V matrices are learned. The outputs from each head are concatenated and projected to obtain the final output of the multi-head attention:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
Where:
head_i = Attention(Q_i, K_i, V_i) for each head i.
W_O is a learned weight matrix that projects the concatenated outputs.

After the multi-head attention layer, a position-wise feed-forward network is applied to each position separately and identically. It consists of two linear transformations with a ReLU activation in between:
FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
Where:
W_1, W_2 are learned weight matrices.
b_1, b_2 are learned biases.

Layer normalization is applied after both the multi-head attention and the feed-forward network, followed by residual connections:
Z = LayerNorm(X + MultiHead(Q, K, V))
Output = LayerNorm(Z + FFN(Z))

The final output of BERT is a sequence of context-aware token representations. For tasks like classification, the representation corresponding to the special [CLS] token (the first token of the input) is typically used.

During pre-training, BERT uses a masked language model objective, where some percentage of the input tokens are randomly masked, and the model is trained to predict these masked tokens. Given the masked input:
X_masked = [x_1, [MASK], x_3, ..., x_n]

The model tries to predict the original token x_2 from its context using:
P(x_2 | X_masked) = softmax(W * h_i + b)
Where:
h_i is the hidden state corresponding to the masked token.
W and b are learned parameters for the output softmax layer.

In addition to MLM, BERT also uses a next sentence prediction task during pre-training. The model is given pairs of sentences and learns to predict whether the second sentence follows the first in the original text.


Given two sentences, A and B, the input is:
Input = [CLS] A [SEP] B [SEP]

BERT outputs a binary label indicating whether B is the next sentence after A:
P(NSP | [CLS]) = softmax(W * h_[CLS] + b)
