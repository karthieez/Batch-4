import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load and preprocess the data
data = pd.read_csv('News.csv', index_col=0)
data = data.drop(["subject", "date"], axis=1)
data = data.sample(frac=1).reset_index(drop=True)

# Ensure class is of type int
data['class'] = data['class'].astype(int)

# Plot class distribution
sns.countplot(data=data, x='class', order=data['class'].value_counts().index)
plt.title("Class Distribution")
plt.show()

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(
            token.lower() for token in sentence.split()
            if token.lower() not in stop_words
        ))
    return preprocessed_text

# Apply preprocessing
data['text'] = preprocess_text(data['text'].values)

# WordCloud for Real news
real_text = ' '.join(word for word in data['text'][data['class'] == 1])
wordcloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False).generate(real_text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Real News")
plt.show()

# WordCloud for Fake news
fake_text = ' '.join(word for word in data['text'][data['class'] == 0])
wordcloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False).generate(fake_text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Fake News")
plt.show()

# Function to get top N words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Bar chart of top words
common_words = get_top_n_words(data['text'], 20)
df1 = pd.DataFrame(common_words, columns=['Word', 'Count'])
df1.groupby('Word').sum()['Count'].sort_values(ascending=False).plot(
    kind='bar', figsize=(10, 6), xlabel="Top Words", ylabel="Count", title="Top Words Frequency"
)
plt.show()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
print("Logistic Regression - Train Accuracy:", accuracy_score(y_train, lr_model.predict(x_train)))
print("Logistic Regression - Test Accuracy:", accuracy_score(y_test, lr_model.predict(x_test)))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
print("Decision Tree - Train Accuracy:", accuracy_score(y_train, dt_model.predict(x_train)))
print("Decision Tree - Test Accuracy:", accuracy_score(y_test, dt_model.predict(x_test)))

# Confusion Matrix
cm = metrics.confusion_matrix(y_test, dt_model.predict(x_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()
