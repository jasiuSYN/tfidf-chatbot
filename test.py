import unittest
from unittest.mock import patch

from preprocessing import preprocess_text
from chatbot import ChatBot


class TestChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.string_1 = "Podaj definicję scrum."
        cls.string_2 = "Kim jest scrum master?"
        cls.string_3 = "Tutaj nauczysz się jak testować."
        cls.chatbot = ChatBot()

    def test_string_preprocessing(self):
        self.assertEqual(preprocess_text(self.string_1), "podać definicja scrum")
        self.assertEqual(preprocess_text(self.string_2), "scrum master")
        self.assertEqual(
            preprocess_text(self.string_3),
            "nauczyć testować",
        )

    @patch("builtins.input", side_effect=["yes", "y", "tak"])
    def test_positive_input(self, mock_input):
        input_1 = mock_input()
        input_2 = mock_input()
        input_3 = mock_input()
        self.assertTrue(input_1.lower() in self.chatbot.positive_words)
        self.assertTrue(input_2.lower() in self.chatbot.positive_words)
        self.assertTrue(input_3.lower() in self.chatbot.positive_words)

        self.assertFalse(input_1.lower() in self.chatbot.negative_words)
        self.assertFalse(input_2.lower() in self.chatbot.negative_words)
        self.assertFalse(input_3.lower() in self.chatbot.negative_words)

    def test_best_response(self):
        self.assertEqual(
            self.chatbot.best_response(self.string_1), self.chatbot.responses[0]
        )
        self.assertEqual(
            self.chatbot.best_response(self.string_2), self.chatbot.responses[5]
        )
        self.assertEqual(self.chatbot.best_response(self.string_3), False)


if __name__ == "__main__":
    unittest.main()
