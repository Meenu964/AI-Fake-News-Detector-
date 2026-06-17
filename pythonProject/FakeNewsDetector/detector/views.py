import pickle
import re
from django.shortcuts import render
from .models import NewsHistory
from textblob import TextBlob

# Load ML files
model = pickle.load(open("ml_model/model.pkl", "rb"))
vectorizer = pickle.load(open("ml_model/vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def home(request):
    result = None
    confidence = None
    sentiment = ""

    if request.method == "POST":
        news = request.POST.get("news")

        # 1. Text Preprocessing
        cleaned_news = clean_text(news)

        # 2. Sentiment Analysis (Using raw or cleaned text as per your choice)
        analysis = TextBlob(news)
        if analysis.sentiment.polarity > 0:
            sentiment = "Positive 😊"
        elif analysis.sentiment.polarity < 0:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        # 3. Vectorization (Sirf CLEANED text ko transform karein)
        vector = vectorizer.transform([cleaned_news])

        # 4. Prediction & Probability
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        # Debugging ke liye terminal zaroor check karein
        print("--- ML Debugging ---")
        print("Raw Prediction Output:", prediction)
        print("Probabilities [Class 0, Class 1]:", probability)

        confidence = round(max(probability) * 100, 2)

        # 5. Label Mapping (Isko training notebook se verify kar lein)
        if prediction == 0:
            result = "Fake News"
        else:
            result = "Real News"

        # 6. Save to History
        NewsHistory.objects.create(
            news_text=news,
            prediction=result,
            confidence=confidence
        )

    return render(
        request,
        "home.html",
        {
            "result": result,
            "confidence": confidence,
            "sentiment": sentiment
        }
    )


def history(request):
    all_news = NewsHistory.objects.all().order_by('-created_at')
    return render(request, "history.html", {"all_news": all_news})