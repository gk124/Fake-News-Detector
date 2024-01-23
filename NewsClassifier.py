import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from datetime import datetime
import getpass

class NewsClassifier:
    def __init__(self):
        self.true_csv_path = 'True.csv'
        self.fake_csv_path = 'Fake.csv'
        self.vectorizer = CountVectorizer()
        self.classifier = PassiveAggressiveClassifier()
        self.load_and_preprocess_data()
        self.train_classifier()
        self.admin_password = "JACKPOT"  # Replace with your actual admin password

    def load_and_preprocess_data(self):
        self.true_news_df = pd.read_csv(self.true_csv_path)
        self.true_news_df['label'] = 'True'

        self.fake_news_df = pd.read_csv(self.fake_csv_path)
        self.fake_news_df['label'] = 'False'

        self.combined_data = pd.concat([self.true_news_df, self.fake_news_df])

        features = self.combined_data['title'] + ' ' + self.combined_data['text']
        labels = self.combined_data['label']

        self.features_vectorized = self.vectorizer.fit_transform(features)
        self.labels = labels

    def train_classifier(self):
        self.classifier.fit(self.features_vectorized, self.labels)

    def authenticate_admin(self):
        password_attempt = getpass.getpass("Enter admin password: ")
        if password_attempt == self.admin_password:
            return True
        else:
            print("Admin not authenticated.")
            return False

    def update_classifier(self, title, text, correct_label):
        if self.authenticate_admin():
            formatted_date = datetime.now().strftime("%B %d, %Y")

            if correct_label.lower() == 'true':
                new_entry = pd.DataFrame({'title': [title], 'text': [text], 'label': ['True'], 'date': [formatted_date]})
                self.true_news_df = pd.concat([self.true_news_df, new_entry], ignore_index=True)
            elif correct_label.lower() == 'false':
                new_entry = pd.DataFrame({'title': [title], 'text': [text], 'label': ['False'], 'date': [formatted_date]})
                self.fake_news_df = pd.concat([self.fake_news_df, new_entry], ignore_index=True)
            else:
                print("Invalid input. Entry not added.")

            # Update the combined data and retrain the classifier with the updated dataset
            self.combined_data = pd.concat([self.true_news_df, self.fake_news_df])
            features = self.combined_data['title'] + ' ' + self.combined_data['text']
            labels = self.combined_data['label']
            self.features_vectorized = self.vectorizer.fit_transform(features)
            self.labels = labels
            self.train_classifier()

            # Save the updated CSV files
            self.true_news_df.to_csv(self.true_csv_path, index=False)
            self.fake_news_df.to_csv(self.fake_csv_path, index=False)
            print("Entry added and classifier retrained.")
        else:
            print("Classifier not retrained.")

    def predict_truthfulness(self, title, text):
        new_news = [title + ' ' + text]
        new_news_vectorized = self.vectorizer.transform(new_news)
        prediction_new_news = self.classifier.predict(new_news_vectorized)[0]

        print(f"Title: {title}")
        print(f"Text: {text}")
        print(f"Predicted label: {prediction_new_news}")

        # Prompt user for feedback
        feedback = input("Was the prediction correct? (y/n): ")
        if feedback.lower() == 'n':
            correct_label = input("Enter the correct label (True/False): ")
            self.update_classifier(title, text, correct_label)
        else:
            print("Classifier not retrained.")


if __name__ == "__main__":
    classifier = NewsClassifier()

    # Loop to continuously prompt for input until 'exit' is entered
    while True:
        title = input("Enter the title of the news (type 'exit' to quit): ")
        if title.lower() == 'exit':
            break

        text = input("Enter the text of the news: ")

        classifier.predict_truthfulness(title, text)
