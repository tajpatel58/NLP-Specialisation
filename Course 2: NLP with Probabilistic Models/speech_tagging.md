# Speech Tagging:

- Speech tagging is the process of ascribing "noun", "adverb", "adjective", "pronoun", "punctuation" etc to words in a corpus. 
- We need this as we can use POS (part of speech tagging) to disambiguate words in sentences.
- For example: 
    - "You did really well on that test"
    - "I found money at the bottom of a well"
    - The word "well" has 2 different use cases, the first being an adjective, the second being a noun. 
- By disambiguating the tag of the word, this can help to improve language model performance.
- Here we will learn how we can assign tags to an entire sentence using the Viterbi algorithm.


## Methodology: 
- The idea is we will use Markov Processes to determine the state of a word, where the states are types of words: adjective, noun, pronoun, verb, advert etc.
    - Recall, the idea behind a Markov chain is the next state given the current/previous states, is only dependent on the current state. 
    - Intuitavely, this says whether a word is adjective, noun, etc is dependent only on the type of the previous word. 
- This is clearly a supervised model in the sense we neeed some training data that is already tagged. We then apply the following steps:

### Notation: 

- Set of Tags: $\{ t_1, t_2, t_3, ..., t_n \}$
- Training Corpus of Tagged Words: $\{ w_1, w_2, w_3, ...., w_m \}$

### Step 1: Transition Matrix:
- We can create the transition matrix which consists of the probabilities of jumping from one state to another. 
- We do this by counting the number of times one state (j) is followed by the state (i) and dividing this by the total number of times state (i) is present. 
- One slight error with this, is if one pair of state doesn't occur then we wouldn't want the probability to be 0. (as it's like you can place any 2 words together in a sentence). To this end, we apply a smoothing hyper parameter to ensure the probability isn't zero:

- <img src="./graphics/pos_transition_smoothing.png" width="700"/>

- Note: N is the number of unique tags not the total count of tags. 
- Note: Transition Matrix is an $(n$ x $n)$ matrix.
- Note: In some cases will be $((n+1)$ x $n)$ matrix, where the initial row encodes the initial distribution. Which we calculate as the starting distribution given by the first word of each training datapoint. 

### Step 2: Emission Matrix:
- The model is built around the idea we want to be able to predict the state of the next word given the current word. We don't want to predict the next word given the current word. 
- To this end, we create another matrix called the Emission Matrix which is an $((n+1)$ x $m)$ matrix.
- Which we think each entry as the probability a word "emits" from a state. Which more formally is the proportion of the number of times a word is in a particular state, divided by the total (non unique) count of words in that state. 
- Similarly to the transition matrix, we apply a smoothing factor, allowing a word to be emitted from any state: 

- <img src="./graphics/pos_emission_smoothing.png" width="700"/>


### Step 3: Hidden Markov Model: 

- Combining the above and excluding smoothing, we've essentially created the below model:

- <img src="./graphics/hidden_markov_model.png" width="700"/>

- It's called the Hidden Markov Model as the states are hidden in the sense of predicted word state to another word state implicitly uses the transition states to move around. 
- The Transition and Emission matrices can be used to label each of the arrows in the HMM. 
- Using this model, we can jump from word to word, attempting to pick the path that generates the highest probability of being picked. 
- In the above HMM, suppose we have the sentence: "I Love to Learn". 
    - The below outlines 2 paths that we can take. Note the brackets contain the state we jump to:
    - I(O) -> love(NN) -> to(O) -> learn(VB)
    - Predicted Tags: O, NN, O, VB
    - I(O) -> love(VB) -> to(O) -> learn(VB)
    - Predicted Tas: O, VB, O, VB
    - The first being slightly incorrect as "love" is a verb in this sentence. 
- Our prediction method is to pick the path that gives the highest probability. Note we multiply the transition and emission probabilities when going between words. 
To this end, we will go through the Viterbi algorithm, which is a graph algorithm which helps us find the path with the higest probability. 