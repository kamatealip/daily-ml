


class OneHotEncoding():
    def __init__(self,data) -> None:
        self.data = data
    
    def corpus(self):
        corpus = list()
        for sentence in self.data:
            for word in sentence.split():
                corpus.append(word)
        return corpus
    
    def vocabulary(self):
        
        return list(set(self.corpus()))

data = ['people watch youtube', 'youtube watch people','people comment on youtube', 'youtube comment on people']

one_hot = OneHotEncoding(data)
print(one_hot.data)

print("Corpus:", one_hot.corpus())
print("Vocabulary:", one_hot.vocabulary())