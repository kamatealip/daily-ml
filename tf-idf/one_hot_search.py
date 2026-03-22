from one_hot_encoding import OneHotEncoding


class OneHotSearch(OneHotEncoding):
    def cosine_similarity(self, sentence_a, sentence_b):
        vector_a = self.multi_hot_vector(sentence_a)
        vector_b = self.multi_hot_vector(sentence_b)

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

    def recommend(self, liked_sentence, top_k=2):
        results = []

        for sentence in self.data:
            if sentence == liked_sentence:
                continue

            score = self.cosine_similarity(liked_sentence, sentence)
            results.append((sentence, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    data = [
        "python basics for beginners",
        "advanced python for data analysis",
        "javascript for web development",
        "machine learning with python",
        "data science projects using python",
    ]

    search_engine = OneHotSearch(data)

    query = "python for data science"
    print("Vocabulary:", search_engine.vocabulary())
    print()
    print("Simple search example using one-hot encoding")
    print("Query:", query)

    for rank, (sentence, score) in enumerate(search_engine.search(query), start=1):
        print(f"{rank}. {sentence} -> similarity score: {score:.3f}")

    liked_course = "machine learning with python"
    print()
    print("Simple recommendation example using one-hot encoding")
    print("If a user likes:", liked_course)

    for rank, (sentence, score) in enumerate(
        search_engine.recommend(liked_course), start=1
    ):
        print(f"{rank}. {sentence} -> similarity score: {score:.3f}")
