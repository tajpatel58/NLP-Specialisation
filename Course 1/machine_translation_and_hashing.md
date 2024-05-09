# Machine Translation + Hashing:

- We saw our vector based models will require a fast lookup to understand which word embeddings are closest to a particular vector. 
- Here we will learn about transforming vectors, locality sensitive hashing and K-Nearest Neighbours. 

## Transforming Vectors: 

- Consider the use case of converting english words to french words. How do we do it: 
    - We generate word embeddings for words in both languages and store them.
    - Then construct a matrix that can map the english words -> french words. 
    - Given a new english word, apply this transformation matrix to get a word embedding in French. 
    - Find the nearest word embedding corresponding to an actual french word.
- Note: you still need to have word embeddings for all possible words, but can train on a subset. 
- So how do we create this matrix: We essentially do a gradient descent. 
    - Initialise a random matrix R as our transformation. 
    - Calculate the predicted output (XR) and work out the Frobenius Norm between the predicted output (which is a matrix of word embeddings), and the actual word embeddings. 
    - The dimensions of this matrix are trivial: we want the same number of columns as the dimension of our output word embedding, and number of rows to be dimension of input word embeddings. Thus if (m) = input dimension, (n) dimension of output embeddings, then R is mxn. 
    - Improve the matrix R by using gradient descent. 

    <img src="./graphics/gradient_descent_matrices.png" width="700"/>