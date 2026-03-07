text = [
    'John likes to watch movies. Mary likes movies too.',
    "mary also likes to watch football games.",
]

ex = "John likes to watch movies. Mary likes movies too. Mary also likes to watch football games."

def bag_of_words(text):
    bag = {}
    for row in text:
        for word in row.split():
            word = word.lower().strip(".,!?;:")
            bag[word] = bag.get(word, 0) + 1
    return bag, set(bag.keys())

result,unique_tokens = bag_of_words(text)

print("Bag of words: ", result)
print("Unique Tokens[vocab words]: ", len(unique_tokens))