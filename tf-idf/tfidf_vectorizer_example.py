from tfidf_vectorizer import TFIDFVectorizer


def print_vector_details(label, text, vectorizer):
    vector = vectorizer.vectorize(text)
    rounded_vector = [round(value, 3) for value in vector]

    print(label)
    print(f"Text: {text}")
    print("TF-IDF vector:", rounded_vector)
    print("Top terms:", vectorizer.top_terms(text))
    print()


if __name__ == "__main__":
    documents = [
        "Python makes machine learning experiments easier to build.",
        "Data science projects often rely on Python and clean datasets.",
        "TF-IDF helps highlight words that matter in each document.",
        "Cricket and football are popular sports around the world.",
    ]

    vectorizer = TFIDFVectorizer(documents)

    print("Vocabulary:", vectorizer.vocabulary())
    print()

    print("IDF scores:")
    for token, score in vectorizer.idf_scores().items():
        print(f"{token}: {score:.3f}")

    print()
    print("Corpus examples:")
    print()

    for index, document in enumerate(documents, start=1):
        print_vector_details(f"Document {index}", document, vectorizer)

    query = "Python data projects"
    print_vector_details("New text example", query, vectorizer)
