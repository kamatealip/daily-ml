from ngram_encoding import NGramEncoding


class NGramSearch(NGramEncoding):
    def cosine_similarity(self, sentence_a, sentence_b):
        vector_a = self.vectorize(sentence_a)
        vector_b = self.vectorize(sentence_b)

        dot_product = 0
        for value_a, value_b in zip(vector_a, vector_b):
            dot_product += value_a * value_b

        magnitude_a = sum(value * value for value in vector_a) ** 0.5
        magnitude_b = sum(value * value for value in vector_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def search(self, query, top_k=3):
        results = []

        for sentence in self.data:
            score = self.cosine_similarity(query, sentence)
            results.append((sentence, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    data = [
        "people watch youtube videos to learn coding",
        "people watch youtube shorts for quick entertainment",
        "youtube videos help people learn python programming",
        "short videos are useful for mobile learning",
    ]

    query = "learn python from youtube videos"
    search_engine = NGramSearch(data, n=2)

    print("N value:", search_engine.n)
    print("Bi-gram vocabulary:", search_engine.vocabulary())
    print()
    print("Practical use: search the most relevant sentence for a query")
    print("Query:", query)

    for rank, (sentence, score) in enumerate(search_engine.search(query), start=1):
        print(f"{rank}. {sentence} -> similarity score: {score:.3f}")
