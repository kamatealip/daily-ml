from one_hot_encoding import OneHotEncoding


class OneHotSearch(OneHotEncoding):
    STOPWORDS = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "using",
        "with",
    }

    def _tokenize(self, sentence):
        tokens = super()._tokenize(sentence)
        return [token for token in tokens if token not in self.STOPWORDS]

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

    def _rank_sentences(self, source_text, top_k=3, skip_sentence=None):
        results = []

        for sentence in self.data:
            if sentence == skip_sentence:
                continue

            score = self.cosine_similarity(source_text, sentence)
            if score > 0:
                results.append((sentence, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def search(self, query, top_k=3):
        return self._rank_sentences(query, top_k=top_k)

    def recommend(self, liked_sentence, top_k=2):
        return self._rank_sentences(
            liked_sentence, top_k=top_k, skip_sentence=liked_sentence
        )

    def recommend_from_query(self, query, top_k=3):
        return self._rank_sentences(query, top_k=top_k)


def print_results(results):
    if not results:
        print("No matching results found.")
        return

    for rank, (sentence, score) in enumerate(results, start=1):
        print(f"{rank}. {sentence} -> similarity score: {score:.3f}")


def get_operation():
    print("Choose an operation:")
    print("1. Search")
    print("2. Recommend")
    choice = input("Enter 1 or 2: ").strip().lower()

    if choice in {"1", "search", "s"}:
        return "search"
    if choice in {"2", "recommend", "recommendation", "r"}:
        return "recommend"
    return None


def get_recommendation_input(search_engine):
    print("Available items:")
    for index, sentence in enumerate(search_engine.data, start=1):
        print(f"{index}. {sentence}")

    user_input = input(
        "Enter an item number or type a topic like python or javascript: "
    ).strip()

    if user_input.isdigit():
        selected_index = int(user_input) - 1
        if 0 <= selected_index < len(search_engine.data):
            return ("item", search_engine.data[selected_index])
        return None

    normalized_input = " ".join(search_engine._tokenize(user_input))
    if not normalized_input:
        return None

    for sentence in search_engine.data:
        if normalized_input == " ".join(search_engine._tokenize(sentence)):
            return ("item", sentence)

    return ("query", user_input)


if __name__ == "__main__":
    data = [
        "python basics for beginners",
        "advanced python for data analysis",
        "machine learning with python",
        "data science projects using python",
        "javascript for web development",
        "frontend projects with javascript",
        "react and javascript user interface design",
        "node javascript backend development",
    ]

    search_engine = OneHotSearch(data)
    print("Vocabulary:", search_engine.vocabulary())
    print()

    while True:
        operation = get_operation()
        print()

        if operation == "search":
            query = input("Enter your search query: ").strip()
            print()
            print("Search results:")
            print_results(search_engine.search(query))
        elif operation == "recommend":
            recommendation_input = get_recommendation_input(search_engine)
            print()

            if recommendation_input is None:
                print("Could not find a matching item for recommendation.")
            else:
                input_type, value = recommendation_input

                if input_type == "item":
                    print("Recommendations based on:", value)
                    print_results(search_engine.recommend(value, top_k=3))
                else:
                    print("Recommendations for topic:", value)
                    print_results(search_engine.recommend_from_query(value, top_k=3))
        else:
            print("Invalid choice. Please select search or recommend.")

        print()
        continue_choice = input("Do you want to perform another operation? (y/n): ")
        if continue_choice.strip().lower() != "y":
            break
        print()
