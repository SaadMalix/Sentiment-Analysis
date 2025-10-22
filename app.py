from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize Flask
app = Flask(__name__)

# Download required NLTK data
nltk.download('vader_lexicon')

# Initialize analyzers
sia = SentimentIntensityAnalyzer()

# Extend VADER lexicon for better accuracy
sia.lexicon.update({
    "mad": -2.0,
    "furious": -3.0,
    "upset": -1.8,
    "angry": -2.5,
    "disappointed": -1.7,
    "happy": 2.5,
    "awesome": 3.0,
    "love": 3.2,
    "sad": -2.2,
    "great": 2.8,
    "terrible": -3.1
})

# Custom slang and typo corrections
CUSTOM_FIXES = {
    "madd": "mad",
    "luv": "love",
    "happyy": "happy",
    "angrry": "angry",
    "goood": "good",
    "badd": "bad",
    "awsm": "awesome",
    "thx": "thanks",
    "gr8": "great"
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']

    if not text.strip():
        return render_template('index.html', result="Please enter some text!")

    # --- Step 1: Apply custom slang and typo fixes ---
    words = text.split()
    fixed_text = " ".join([CUSTOM_FIXES.get(w.lower(), w) for w in words])

    # --- Step 2: Apply spell correction using TextBlob ---
    blob = TextBlob(fixed_text)
    corrected_text = str(blob.correct())

    # --- Step 3: Get sentiment from TextBlob ---
    blob_score = blob.sentiment.polarity  # Range: -1 to +1

    # --- Step 4: Get sentiment from VADER ---
    vader_score = sia.polarity_scores(corrected_text)['compound']

    # --- Step 5: Combine both (average) ---
    combined_score = (blob_score + vader_score) / 2

    # --- Step 6: Classify sentiment ---
    if combined_score > 0.1:
        sentiment = "😊 Positive"
    elif combined_score < -0.1:
        sentiment = "😔 Negative"
    else:
        sentiment = "😐 Neutral"

    return render_template(
        'index.html',
        text=text,
        corrected=corrected_text,
        result=sentiment
    )


if __name__ == '__main__':
    app.run(debug=True)
