import streamlit as st 
import nltk
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import speech_recognition as sr
import pyttsx3
import requests

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load BioGPT model and tokenizer
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
set_seed(42)

# Initialize NLP components
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer()

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words("english")]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)

def fetch_drug_info(medication):
    url = f"https://api.fda.gov/drug/label.json?search={medication}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['results'][0]['description'] if 'results' in data else "No drug information found."
    return "Failed to fetch drug information."

def healthcare_chatbot(user_input):
    processed_input = preprocess_text(user_input)
    if "symptom" in processed_input:
        return "Please consult a doctor for accurate advice."
    elif "appointment" in processed_input:
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in processed_input:
        medication_name = user_input.split()[-1]  # Extract the last word as medication name
        return fetch_drug_info(medication_name)
    else:
        response = chatbot(processed_input, max_length=100, num_return_sequences=1, do_sample=True)
        return response[0]['generated_text']

def main():
    st.title("Healthcare Assistant Chatbot")
    st.write("Hello! I'm here to assist you with your healthcare-related queries.")
    
    # Voice input button
    if st.button("🎤 Speak"):
        with sr.Microphone() as source:
            st.write("Listening...")
            try:
                audio = recognizer.listen(source)
                user_input = recognizer.recognize_google(audio)
                st.write(f"Recognized: {user_input}")
            except sr.UnknownValueError:
                st.write("Sorry, could not understand your speech.")
                return
            except sr.RequestError:
                st.write("Could not request results, please check your connection.")
                return
    else:
        user_input = st.text_input("How can I assist you today?", placeholder="Type your question here...")
    
    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
                st.success("Healthcare Assistant:")
                st.write(response)
                
                # Convert response to speech
                engine.say(response)
                engine.runAndWait()
        else:
            st.warning("Please enter a message to get a response.")
    
    st.markdown("---")
    st.write("*Note: This chatbot provides general information and does not replace professional medical advice.*")

if __name__ == "__main__":
    main()

