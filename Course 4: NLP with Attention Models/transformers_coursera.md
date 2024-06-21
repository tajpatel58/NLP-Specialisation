# Transformers: 
- Transformers is another kind attention based sequence to sequence model. 
- It is a sequence model, in the sense it takes input as a sequence and outputs a sequence. 

## Transformers vs RNNs:

### Parallel Computation: 
- We recall that in RNNs and the original Seq2Seq model (which utilised LSTMs) the hidden states had to be calculated sequentially for the inputs to then be used as input to the next set of LSTMs/unit. We would repeat this until the end of the sequence to obtain the encoded version of the input sequence. 
- Thus, if we had a long sequence of words, it would take longer and longer to compute the encoder output. 
- This method doesn't allow for parallel computation, whereas we've seen that the transformer mechanism allows for parallel computation of words, keys and values, and therefore self attention can be computed much quicker and finally the encoded vectors of the input sequence can be calculated quickly. 


### Loss of Information: 
- As the original seq2seq model wasn't attention based, key words in the start of the sequence may have gotten lost by the final encoded vector of the input sequence. 
- Transformers don't have this problem due to the self attention mechanism and the encoder-decoder attention mechanism. 
- Due to the vanishing gradient problem, gradients can become really small and hence your model will not learn much from the starting terms. 

### Position of a Word:
- RNNs carry knowledge of position of a word as it applies weights in a particular sequence.
- Transformers will use positional encoding. 

## Transformer Applications: 
- Text Summarisation
- Auto Complete
- Named Entity Recognition
- Q&A
- Translation (Machine Neural Translation)
- Chatbots

## Modern Day Transformers:
- GPT: Generative Pretraining Transformer
- BERT: Bidirectional Encoder Representations from Transformers
- T5: Text to Text Transformer

### T5: Text to Test Transformer:
- T5 can do a bunch of tasks without requiring retraining/adjustment. (just like ChatGPT)
- Classification (is a piece of text making sense?)
- Question & Answer (ask a question, return an answer)
- Machine Translation (translate a language)
- Summarisation. 