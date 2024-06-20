# Transformers: 

- Transformers is another advancement on Seq2Seq models. 
- Tools like ChatGPT run Transformers in the background. 
- We'll go through some comparisons with RNNs. 


## Word Embeddings + Positional Encoding:
- Like most of our models in NLP, our sentences go through an embedding layer. 
- However, one thing that is different here, is we add a "positonal encoding" layer.
- There are many ways to positional encode but one way is to add a vector to the word embedding of a word, based on where that word is in the sequence. This vector that we add is based on some sine/cosine values that are fixed. This means the same word will have a different positional encoding (and thus a different word embedding) depending on where it is in the sequence. 
- For the vector that we're adding for positional encoding, it is made up of different values from sine/cosine curves. Specifically, each element in the vector relates to a different sine/cosine curve. (from this we infer if our word embedding is $n$ dimensional, we have $n$ curves) We can imagine this curve having "word token position in input" on the x-axis, and therefore for each position we have a sine/cosine value.
- Due to periodicity for different position inputs, may generate the same value, but the other elements will have different sine/cosine values for meaning the overall vector will be different. Moral of the story: **each input word will have a unique positional encoding vector**
-  <img src="./graphics/positional_encoding.png" width="1000"/>


## Attention: 
- The seq2seq model, that was built around LSTMs also used a form of attention, where we:
    - Calculated the similarity between the hidden states of in our encoder with a hidden state in the decoder.
    - Applied softmax to the similarity scores.
    - Took a weighted average of the hidden states in the encoder and fed into the dense layer. 
    - Here the query(Q): decoder hidden state, keys(K): encoder hidden states, values(V): encoder hidden states. 

### Self Attention:
- Transformers use an alternative mechanism to attention, (from what I can see, I don't think transformers use LSTMs). 
- The intuitive idea is the same as before, we want to emphasise on important words within the sentence. However in this case, we also apply the attention mechanism to the words in the input sequence. Ie: We alter the encoding of a word in a sequence based on the other words in the sequence. This is because words in the input sequence may relate to each other and we want the model to pick up on this to adjust the output accordingly. 
- Eg: "I got my car straight from the showroom, it was new", we want "it" to relate more to the word "Car" than "showroom" as that's what it refers to. 
- Suppose our word embedding is $n$ dimensional, have $k$ words, we'll repeat the following for every word embedding: 
    1. Apply 3 dense layers (output dimension = $n$, thus all $(n \text{ x } n)$ matrices):
        - One layer will output the query(Q) for a word.
        - One layer will output the key(K) for a word.
        - One layer will output the value(V) for a word. 
    2. Compare the query of a word with the keys of the remaining words (itself). (ie: workout the similarity scores - one way is dot product). ($k$ similarity scores)
    3. Compute the softmax vector for to represent a probability distribution.
    4. Take the weighted average of the value vectors. 
    5. Output vector is now the attention encoded word embedding, this incorporates the information from the other words. (more from words with higher similarity.) (output vector still $n$ dimensional.)
- The dense layers we apply are the same for every word and thus the queries,keys,values can be calculated for each word in parallel which gives us access to parallel computation. 
- The above describes a "Self Attention" cell, which we can stack to have multiple attention mechanisms for complicated sentences/paragraphs. Each cell will have it's own weights just like how we stacked LSTM cells in the Seq2Seq models. This is called Multi-Head Attention. (original model had 8 stacks of self attention cells)
- Each Self Attention Cell will have $3n^2$ training parameters. 
-  <img src="./graphics/self_attention.png" width="1000"/>

## Residual Connection:
- After an attention cell, we will add the output vector with the input vector to the attention mechanism. 
- This is because the attention mechanism will find it hard to preserve the information from the other words, word embedding and the positional encoding as well as the input. 
- We want attention to focus on relationships between words not necessarily the word itself. 
- Formally: 
$$ v := v_{attention} + v_{attention:input}$$

## Transformer Architecture:

### Encoder:
- Transformers create encodings that are:
    - Word Meaning Representative (include information from the original word embedding).
    - Position Aware (includes information aboute the positional encoding).
    - Encode relationships based on the surrounding words in the sequence. (Self Attention)

### Decoder: 
- The decoding process is similar to what we have seen in the seq2seq model, but does have some key differences. 
- Once we have built our encoder and worked out encodings for each words, we can start to decode. 
- The decoder trainable parameters are mostly independent from the encoder. 
- If doing machine translation, we will have to generate new word embeddings for the new language, if it's the same language we maybe able to get away with the same word embedding BUT NOT the same encodings. 
- We start with an encoded sequence and want to generate a response/sequence. We start by decoding the EOS token, but the general decoding steps are:
    - Feed the input word through a word embedding layer.
    - Feed the word embedding through a positional encoding layer. 
    - Then feed through a self attention cell. (Note: self attention is only applied to the terms on the decoder/encoder side. ie: the first token being decoded will only have similarity with itself, the second token being decoded will only have similarity with itself and the first token etc). This layer again, allows information to be preserved based on what has been generated already. 
    - The output of the attention cell, is then passed into a residual connection (adding with it's own word embedding/positional encoding). 
    - **IMPORTANT:** We then feed our output through a **Encoder-Decoder Attention Mechanism**. This is a cell where we apply attention in a similar way to the seq2seq model. It's to allow information from the input sequence to influence the output generated. To this end we add new trainable parameters:
        - On the encoder side we have 2 new dense layers **applied to the output of residual connection** (again both $n$ output units) that will generate keys(K) and values(V). 
        - On the decoder side, we have a new dense layer also **applied to the output of residual connection** (again, $n$ output units), this will be the queries(k).
        - We then apply an attention mechanism as we have the keys, queries and values, can calculate similarity scores, get the softmax and then take the weighted average to produce another $n$ dimensional vector which encodes the input information, as well as the decoded output information. 
        - Note: we can also stack Encoder/Decoder Attention mechanisms. 
    - As we have done another attention mechanism, we pass through a residual connection, adding the inputs prior to the attention mechanism. 
    - We now have a decoded $n$ dimensional vector and we feed through a fully connected layer to get the probabilities of the next word in the sequence. (output dimension is the size of output vocabulary)
    - We can then apply a few techniques like Beam Search or Minimum Bayes Risk or Greedy to select the next word. 
- We repeat the above until we get an EOS token or if using teacher forcing, until the expected sequence has been fed in. 
- **Encoder-Decoder Attention** gets it's name as it's an attention mechanism that compares the relationship between the decoding values and the input sequence output encoder values. 
- **Self Attention** gets it's name as it's only comparing the current words' relationship between other words in the input sequence. 

### Full Architecture: 
- The below depicts the full architecture a transformer uses.

-  <img src="./graphics/transformer.png" width="1000"/>

