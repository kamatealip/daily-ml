class TermFrequencySearch:
    def __init__(self, lines):
        self.lines = lines
        self._tokenized_lines = [self._tokenize(line) for line in lines]
        self._vocabulary = self._build_vocabulary()
        self._term_frequencies = [
            self._build_term_frequency(tokens) for tokens in self._tokenized_lines
        ]

    def _tokenize(self, text):
        tokens = []
        for word in text.split():
            token = word.lower().strip(".,!?;:")
            if token:
                tokens.append(token)
        return tokens

    def _build_vocabulary(self):
        seen = set()
        vocabulary = []

        for tokens in self._tokenized_lines:
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    vocabulary.append(token)

        return vocabulary

    def _build_term_frequency(self, tokens):
        term_frequency = {}
        total_tokens = len(tokens)

        if total_tokens == 0:
            return term_frequency

        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1

        for token in term_frequency:
            term_frequency[token] /= total_tokens

        return term_frequency

    def vocabulary(self):
        return self._vocabulary.copy()

    def search(self, word):
        token = word.lower().strip(".,!?;:")
        results = []

        for index, line in enumerate(self.lines, start=1):
            score = self._term_frequencies[index - 1].get(token, 0.0)
            if score > 0:
                results.append((index, line, score))

        results.sort(key=lambda item: item[2], reverse=True)
        return results


if __name__ == "__main__":
    lines = [
        "Python makes text processing simple and readable.",
        "This line talks about python and machine learning.",
        "Search engines can find a word in many text lines.",
        "Python examples are helpful for beginners in coding.",
        "Data analysis with python is very popular.",
    ]

    search_engine = TermFrequencySearch(lines)

    print("Vocabulary:", search_engine.vocabulary())
    print()

    search_word = input("Enter a word to search: ").strip()
    print()

    results = search_engine.search(search_word)

    if not results:
        print("No lines contain that word.")
    else:
        print(f"Lines containing '{search_word}':")
        for index, line, score in results:
            print(f"{index}. {line} -> term frequency: {score:.3f}")
