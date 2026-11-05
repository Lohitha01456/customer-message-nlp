import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv("support_messages.csv")

print(data.head())
print("\nRows:", len(data))
print("\nMessages per category:")
print(data["category"].value_counts())



X = data["message"]
y = data["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("Training messages:", len(X_train))
print("Test messages:", len(X_test))





unigram_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000),
    ),
])

unigram_model.fit(X_train, y_train)



unigram_predictions = unigram_model.predict(X_test)

unigram_accuracy = accuracy_score(y_test, unigram_predictions)

print("\nUnigram accuracy:", round(unigram_accuracy, 3))

results = pd.DataFrame({
    "message": X_test,
    "actual": y_test,
    "predicted": unigram_predictions,
})

print("\nTest predictions:")
print(results.to_string(index=False))


bigram_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000),
    ),
])

bigram_model.fit(X_train, y_train)

bigram_predictions = bigram_model.predict(X_test)

bigram_accuracy = accuracy_score(y_test, bigram_predictions)

print("\nUnigram + Bigram accuracy:", round(bigram_accuracy, 3))

wrong_predictions = results[
    results["actual"] != results["predicted"]
]

print("\nIncorrect predictions:")
print(wrong_predictions.to_string(index=False))


new_messages = [
    "Money disappeared from my card after checkout",
    "I cannot get into my account",
    "My parcel should have arrived yesterday",
    "The app closes as soon as I open it",
    "I need my money back",
    "The sign in code is missing",
]

unigram_new = unigram_model.predict(new_messages)
bigram_new = bigram_model.predict(new_messages)

for message, uni, bi in zip(new_messages, unigram_new, bigram_new):
    print("\nMessage:", message)
    print("Unigram prediction:", uni)
    print("Unigram + Bigram prediction:", bi)

    probabilities = bigram_model.predict_proba(new_messages)
classes = bigram_model.classes_

for message, scores in zip(new_messages, probabilities):
    print("\nMessage:", message)

    ranked = sorted(
        zip(classes, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    for category, score in ranked:
        print(category, "=", round(score, 3))

tricky_messages = [
    "I can login but my payment keeps failing",
    "My order arrived but the device does not work",
    "Great, another package that never arrived",
    "payment deduct ayyindi but order confirm avvaledu",
    "I need help with something",
]

tricky_predictions = bigram_model.predict(tricky_messages)

for message, prediction in zip(tricky_messages, tricky_predictions):
    print("\nMessage:", message)
    print("Prediction:", prediction)
