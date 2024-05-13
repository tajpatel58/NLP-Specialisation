# Autocorrect: 

- In this week, we'll look at building an autocorrect system for text. 
- The system we will build will recognise spelling mistakes like "Hapy Birthday" but won't recognise contextual/grammatical errors like "Happy Birthday Deer". 


## Levenshtein Distance: 
- We know that Levenshtein distance is minimum number of steps to transform one word to another. eg: Happpy -> Happy requires 1 step, a deletion of the extra p.
- Intuitavely, we can think of autocorrect as picking the word with the closest levenshtein distance.
- Foramlly, the possible steps are: 
    - Insert (add a letter).
    - Delete (removal of a letter)
    - Switch (swap 2 adjacent letters - ONLY ADJACENT)
    - Replace (change 1 letter to another)
- Formally the autocorrect model we will implement will:
    - Identify a mispelt word. (compare against known words)
    - Find all the strings within n edit distance away. 
    - Filter Candidates by only picking the strings that are actual words from the last step.
    - Calculate word probabilities and choose the word that is most likely to occur. We can also try to choose the word most likely based on context. (though we won't do that here). 
- We can try to measure/gauge context like we did in chapter 1: by looking at the nearby words. 
- Instead, to measure likelihood: we apply a Counter instance to a large body of text, then divide the word frequencies by the total count of words. 
- We will also implement a weighting/cost system to each edit, which will be used to workout the levenshtein distance:
    - insert = 1
    - delete = 1
    - replace = 2 (can be thought of delete + insert)

## Minimum Edit Distance:
- Ideally, we want to find the closest word string to the string we're applying autocorrect on. 
- In some cases 1 edit isn't enough generate a new word, especially when we have long strings. 
- Suppose we phrase our problem as how can we compare 2 strings, well we can workout the smallest numbers of edits needed to take one word to another. 
- We can imagine this is a complex operation when we have lengthy strings. To this end, we go through a dynamic programming approach which will improve the time complexity. 

