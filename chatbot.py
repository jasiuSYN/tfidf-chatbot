from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text


class ChatBot:
    def __init__(self) -> None:

        self.bot_name = "Scrummy"
        self.negative_words = [
            "exit",
            "no",
            "quit",
            "nie",
            "pa",
            "n",
        ]
        self.positive_words = [
            "yes",
            "yep",
            "ye",
            "y",
            "tak",
            "ok",
        ]
        self.responses = None
        self.preprocessed_docs = None
        self.get_responses()
        self.get_preprocessed_docs()

    def get_responses(self):
        with open("responses.txt", "r", encoding="utf-8") as respo:
            responses = respo.read().split("\n\n")
        self.responses = responses

    def get_preprocessed_docs(self):
        docs_list = [preprocess_text(doc) for doc in self.responses]
        self.preprocessed_docs = docs_list

    def best_response(self, user_input):
        # preprocess user input and add to responses list
        self.preprocessed_docs.append(preprocess_text(user_input))

        # create tfidf vectorizer:
        vectorizer = TfidfVectorizer()

        # fit and transform vectorizer on processed docs:
        tfidf_vectors = vectorizer.fit_transform(self.preprocessed_docs)

        # compute cosine similarity betweeen the user message tf-idf vector and the different response tf-idf vectors:
        cosine_similarities = cosine_similarity(tfidf_vectors[-1], tfidf_vectors)

        # end if the similarity between the question and the answers is equal to 0 without last element (user question)
        if sum(cosine_similarities[0][:-1]) == 0.0:
            return ""

        # get the index of the most similar response to the user message:
        similar_response_index = cosine_similarities.argsort()[0][-2]
        best_response = self.responses[similar_response_index]
        print(best_response)

        print(
            "\n** Po więcej informacji wejdź na stronę: **\nhttps://scrumguides.org/docs/scrumguide/v2020/2020-Scrum-Guide-Polish.pdf"
        )
        print("\n-----------------")

        # delete last user input for next iteration if response isnt exhaustive or not adequately specified
        del self.preprocessed_docs[-1]

    def chat(self):
        print(
            f"Cześć nazywam się {self.bot_name}. Odpowiem na wszystkie Twoje pytania dotyczące metodyki SCRUM.\n"
        )

        while True:
            user_question = input("Co chcesz wiedzieć o SCRUM?\n>>")
            if self.best_response(user_question) == "":
                print(
                    "Przepraszam nie rozumiem. Twoje pytanie nie pasuje do żadnej odpowiedzi!"
                )
                continue
            user_message = input("\nCzy odpowiedź jest dla Ciebie wyczerpująca?\n>>")

            if user_message.lower() in self.positive_words:
                user_message = input(f"Super, czy chcesz zadać następne pytanie?\n>>")

                if user_message.lower() in self.negative_words:
                    print(
                        f"Dziękuję za skorzystanie ze {self.bot_name}. Do zobaczenia!"
                    )
                    break
                else:
                    continue

            elif user_message.lower() in self.negative_words:
                print("Doprecyzuj pytanie.")
                continue

            else:
                user_message = input(
                    "Nie rozumiem! Czy odpowiedź była wyczerpująca? (Tak/Nie)\n>>"
                )
                if user_message.lower() in self.positive_words:
                    print(
                        f"Dziękuję za skorzystanie ze {self.bot_name}. Do zobaczenia!"
                    )
                    break


if __name__ == "__main__":
    a = ChatBot()
    a.chat()
