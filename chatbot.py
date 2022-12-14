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
        self.responses = self.get_responses()
        self.preprocessed_docs = self.get_preprocessed_docs()

    def get_responses(self):
        with open("responses.txt", "r", encoding="utf-8") as respo:
            return respo.read().split("\n\n")

    def get_preprocessed_docs(self):
        return [preprocess_text(doc) for doc in self.responses]

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
            return False

        # get the index of the most similar response to the user message:
        similar_response_index = cosine_similarities.argsort()[0][-2]
        best_response = self.responses[similar_response_index]

        # delete last user input for next iteration if response isnt exhaustive or not adequately specified
        del self.preprocessed_docs[-1]
        return best_response

    def chat(self):
        print(
            f"Cze???? nazywam si?? {self.bot_name}. Odpowiem na wszystkie Twoje pytania dotycz??ce metodyki SCRUM.\n"
        )

        while True:
            user_question = input("Co chcesz wiedzie?? o SCRUM?\n>>")
            if user_question in self.negative_words:
                break
            best_response = self.best_response(user_question)
            if not best_response:
                print(
                    "Przepraszam nie rozumiem. Twoje pytanie nie pasuje do ??adnej odpowiedzi!"
                )
                continue
            else:
                print(best_response)
                print(
                    "\n** Po wi??cej informacji wejd?? na stron??: **\nhttps://scrumguides.org/docs/scrumguide/v2020/2020-Scrum-Guide-Polish.pdf"
                )

                print("\n-----------------")
            user_message = input("\nCzy odpowied?? jest dla Ciebie wyczerpuj??ca?\n>>")

            if user_message.lower() in self.positive_words:
                user_message = input(f"Super, czy chcesz zada?? nast??pne pytanie?\n>>")

                if user_message.lower() in self.negative_words:
                    print(
                        f"Dzi??kuj?? za skorzystanie ze {self.bot_name}. Do zobaczenia!"
                    )
                    break
                else:
                    continue

            elif user_message.lower() in self.negative_words:
                print("Doprecyzuj pytanie.")
                continue

            else:
                user_message = input(
                    "Nie rozumiem! Czy odpowied?? by??a wyczerpuj??ca? (Tak/Nie)\n>>"
                )
                if user_message.lower() in self.positive_words:
                    print(
                        f"Dzi??kuj?? za skorzystanie ze {self.bot_name}. Do zobaczenia!"
                    )
                    break


if __name__ == "__main__":
    a = ChatBot()
    a.chat()
