import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
def calculate_word_frequency(reviews):
    all_reviews_text = ' '.join(reviews)
    words = word_tokenize(all_reviews_text)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    frequency_distribution = FreqDist(words)
    return frequency_distribution
if __name__ == "__main__":
    customer_reviews = [
        "The product is amazing! I love it.",
        "Not satisfied with the quality. Would not recommend.",
        "Fast shipping and excellent customer service.",
        "This is the best product ever!"
    ]
    word_frequency = calculate_word_frequency(customer_reviews)
    print("Top 10 words and their frequencies:")
    for word, frequency in word_frequency.most_common(10):
        print(f"{word}: {frequency}")
