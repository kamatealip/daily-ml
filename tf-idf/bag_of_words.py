text = [
    'John likes to watch movies. Mary likes movies too.',
    "mary also likes to watch football games.",
]

ex = "John likes to watch movies. Mary likes movies too. Mary also likes to watch football games."

def tokenize(sentence):
    words = []
    for w in sentence.split():
        w = w.lower().strip(".,!?;:")
        words.append(w)
    return words

def bag_of_words_vector(text):
    docs = [tokenize(sentence) for sentence in text]
    
    # building vocab 
    vocab = sorted(set(word for doc in docs for word in doc))
    
    vectors = []
    
    for doc in docs:
        vec = []
        for word in vocab:
            vec.append(doc.count(word))
        vectors.append(vec)
        
    return vocab, vectors

vocab, vectors = bag_of_words_vector(text)

print("Vocabulary: ", vocab)
print()

for i, v in enumerate(vectors):
    print(f"Doc {i + 1} vector: ", v)