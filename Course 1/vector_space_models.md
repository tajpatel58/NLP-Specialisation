# Vector Space Models: 

- Vector space models are useful as we can use linear algebra to compare words. 
- The idea behind vector space models is to represent words as vectors. (called word embeddings)
- These vector representations would ideally be generated using context, ie: words that are similar in meaning will be closer. eg: adjectives, verbs, nouns. 

## Basic Word Embedding Generation: Word by Word:

- We specify an integer value of k, we then create a square (NxN) matrix where N = number of words in text. (This is called a co-occurence matrix).
- Each row, column corresponds to a word-word combination, and the value of the matrix at this point is the number of times word_1 is within k (distance/words) of word_2. 

 <img src="./graphics/word_by_word.png" width="700"/>

 - The vector representation of the word "simple" is basically the 1d-vector (2). 
 - Note the column of this co-occurence matrix is what form the word embeddings. 

 ## Basic Word Embedding Generation: Word by Doc:
 
 - Here we create a word embedding by looking at different corpusses containing the word. 
 - Similarly, we create a matrix with all words in the vocabulary, the rows are the words and the columns are the frequencies of each word in a variety of corpuses and each column is a word embedding for the cateoory.  Eg in the screenshot: Economy = (6620, 4000)

 <img src="./graphics/word_by_doc.png" width="700"/>


 ## Vector Similarity: 
 - The basic way of measuring distance between 2 vectors is to use euclidean distance.
 - The second is the cosine similarity. Imagine we had 2 corpuses, and 1 small corpus. In the small corpus, all the words are bound to have a lower occurence, as there are just simply fewer words which will result in the Euclidean distance being higher than usual. 
 - To account for this: the cosine similarity is used to measure the angle between 2 vectors. 