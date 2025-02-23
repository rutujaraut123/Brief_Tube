from flask import Flask, render_template, request, make_response
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from youtube_transcript_api import YouTubeTranscriptApi
import re
import nltk
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, cmudict
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
import time
import traceback

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('cmudict')

app = Flask(__name__)
d = cmudict.dict()
translator = Translator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/get_transcript', methods=['POST'])
def get_transcript():
    link = request.form['link']
    video_id = extract_video_id(link)

    try:
        hindi_transcript = get_transcript_text(video_id, 'hi')
        highlighted_transcript = highlight_keywords(hindi_transcript)  # Highlight here
        return render_template('transcript.html', transcript=highlighted_transcript)
    except Exception:
        try:
            english_transcript = get_transcript_text(video_id, 'en')
            highlighted_transcript = highlight_keywords(english_transcript)  # Highlight here
            return render_template('transcript.html', transcript=highlighted_transcript)
        except Exception:
            return render_template('error.html', message="Error retrieving transcript: Could not find a valid transcript in Hindi or English.")

@app.route('/translate_to_marathi', methods=['POST'])
def translate_to_marathi():
    return translate_text(request, 'mr')

@app.route('/translate_to_hindi', methods=['POST'])
def translate_to_hindi():
    return translate_text(request, 'hi')

@app.route('/translate_to_english', methods=['POST'])
def translate_to_english():
    return translate_text(request, 'en')

def translate_text(req, language):
    transcript = req.form['transcript']
    try:
        translated_transcript = translate_text_chunks(transcript, language)
        return render_template('transcript.html', transcript=transcript, translated_transcript=translated_transcript)
    except Exception as e:
        print(f"Translation error: {e}")
        traceback.print_exc()
        return render_template('error.html', message="Translation failed. Please check the logs.")

def translate_text_chunks(text, language, chunk_size=250, initial_delay=0.5, max_retries=3):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []
    for chunk in chunks:
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                translated_chunks.append(translator.translate(chunk, dest=language).text)
                time.sleep(delay)
                break
            except Exception as e:
                print(f"Error translating chunk: {e}, retry {retries + 1}")
                retries += 1
                delay *= 2
                time.sleep(delay)
        if retries == max_retries:
            translated_chunks.append("Translation Error")
    return "".join(translated_chunks)

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    text_to_convert = request.form.get('transcript') or request.form.get('summary')
    if not text_to_convert:
        return render_template('error.html', message="No transcript or summary provided for PDF generation.")
    pdf_content = generate_pdf_reportlab(text_to_convert)
    response = make_response(pdf_content)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=document.pdf'
    return response

def generate_pdf_reportlab(text):
    font_path = "C:/BreifTube/BreifTube/NotoSansDevanagariUI-Regular.ttf"
    pdfmetrics.registerFont(TTFont('NotoSansDevanagari', font_path))
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    styles['BodyText'].fontName = 'NotoSansDevanagari'
    elements = [Paragraph(line, styles["BodyText"]) for line in text.split('\n') if line]
    doc.build(elements)
    pdf_content = buffer.getvalue()
    buffer.close()
    return pdf_content

def extract_video_id(link):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', link)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")

def get_transcript_text(video_id, language):
    transcript_obj = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
    transcript = " ".join([sub["text"] for sub in transcript_obj])
    return clean_transcript(transcript)

def clean_transcript(transcript):
    transcript = re.sub(r'\[.*?\]', '', transcript)
    transcript = re.sub(r'\s+', ' ', transcript).strip()
    return transcript

def count_syllables(word):
    word = word.lower()
    if word in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        return len(re.findall(r'[aeiouy]+', word))
# Highlight complex words based on syllables
def highlight_keywords(summary, syllable_threshold=3):
    # Tokenize words and get POS tags
    words = word_tokenize(summary)
    pos_tags = pos_tag(words)

    # Use a set of English stopwords to avoid highlighting common words
    stop_words = set(stopwords.words('english'))

    highlighted_keywords = set()

    # Helper function to highlight a keyword if it's complex (more than syllable_threshold syllables)
    def highlight_keyword(match):
        keyword = match.group(0).lower()
        syllable_count = count_syllables(keyword)

        # Highlight words that are not stopwords and exceed the syllable threshold
        if keyword not in stop_words and syllable_count >= syllable_threshold and keyword not in highlighted_keywords:
            highlighted_keywords.add(keyword)
            # Return the word as a clickable link for Google search
            return f'<a href="https://www.google.com/search?q={keyword}" target="_blank" class="highlight">{keyword}</a>'
        return match.group(0)

    # Use regex to find and highlight words
    summary = re.sub(r'\b(\w+)\b', highlight_keyword, summary, flags=re.IGNORECASE)
    return summary


@app.route('/get_summary', methods=['POST'])
def get_summary():
    transcript = request.form['transcript']
    summary = generate_summary(transcript)
    highlighted_summary = highlight_keywords(summary)
    return render_template('transcript.html', summary=highlighted_summary)

def generate_summary(transcript_text):
    sentences = re.split(r'(?<=\w[.!?])\s+', transcript_text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = {sentence: tfidf_matrix[idx].sum() for idx, sentence in enumerate(sentences)}
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = ' '.join(ranked_sentences[:3])
    return summary

if __name__ == '__main__':
    app.run(debug=True)