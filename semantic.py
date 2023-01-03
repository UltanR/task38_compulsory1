import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# It makes sense that 'cat' and 'monkey' are the most similar, both being animals.
# It also makes sense that 'monkey' and 'banana' are similar, being conceptually from the same place
# I did expect all the values to be higher, because they are all nouns, and all biological entities

# Below; other code extracts from pdf
tokens = nlp('cat apple monkey banana ')

for token1 in tokens:

    for token2 in tokens:

        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:

    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


# My own examples
word3 = nlp("oak")
word4 = nlp("philosophy")
word5 = nlp("bodacious")

print(word3.similarity(word4))
print(word4.similarity(word5))
print(word3.similarity(word5))
print()

# In the example for word3, word4, and word5, I was surprised; I expected all values to be low, but apparently
# 'philosophy' and 'bodacioua' have a similarity of 0.32, and even weirder 'oak' and 'bodacious' have a similarity 
# of 0.28

nlp_ = spacy.load('en_core_web_sm')

word1 = nlp_("cat")
word2 = nlp_("monkey")
word3 = nlp_("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Using the simplified language model, the values are much higher. This might have to do with the fact that this model 
# no word vectors loaded, leading to biased results