import streamlit as st 
import nltk
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")
nltk.download("stopwords")

# Load the BioGPT model and tokenizer
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)


def healthcare_chatbot(user_input):
    if "symptom" in user_input:
        return "Please Consult Doctor For Accurate Advice"
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with the Doctor?"
    elif "medication" in user_input:
        return "It's important to take prescribed medicine regularly. If you have any concerns, consult your doctor."
    else:
        # Use the new BioGPT model for general responses
        response = generator(user_input, max_length=500, num_return_sequences=1, do_sample=True)
        return response[0]['generated_text']

def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How Can I Assist You Today?")
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
                st.write("Healthcare Assistant: ", response)
        else:
            st.write("Please enter a message to get a response.")

if __name__ == "__main__":
    main()
