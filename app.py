import streamlit as st
import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up OpenAI API key
openai.api_key = 'your-openai-api-key'

# Load CSV data using the new caching method
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')  # Ensure the CSV is in the correct location
    return df

# Function to find the most similar question from the CSV
def find_similar_question(user_input, df):
    questions = df['issue'].tolist()

    # Use TF-IDF to find the closest question
    vectorizer = TfidfVectorizer().fit_transform([user_input] + questions)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between user input and the questions in the CSV
    cosine_sim = cosine_similarity([vectors[0]], vectors[1:])
    
    # Get the index of the most similar question
    similar_idx = cosine_sim.argsort()[0][-1]
    return df.iloc[similar_idx]['solution']

# Function to get a fallback response from OpenAI
def get_openai_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or use gpt-3.5-turbo if preferred
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps students resolve issues during online exams."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Streamlit app UI
st.set_page_config(page_title="Exemplify Exam Support Bot", page_icon="📝", layout="wide")

# Title and introduction
st.title("📝 Exemplify Exam Support Bot")
st.markdown("""
    #### Welcome to the Exemplify Exam Support Bot! 
    Facing technical issues during exams? Describe your problem below, and we'll provide you with real-time assistance!
    """)

# Load the exam issues CSV data
df = load_data()

# Sidebar for additional information
st.sidebar.header("📌 How It Works")
st.sidebar.markdown("""
1. **Describe Your Issue**: Enter the issue you're facing during the exam.
2. **Get Instant Solution**: We will match your issue with our database or consult GPT if needed.
3. **Professional Help**: We provide tailored solutions to help you complete your exam smoothly.
""")

# User input section
st.write("Describe the issue you're facing during the exam:")

user_issue = st.text_area("Enter the issue here", height=150, placeholder="e.g., My screen froze, Excel is not working, etc.")

# Call to action button
if st.button("🔍 Find Solution"):
    if user_issue:
        with st.spinner('Searching for the best solution...'):
            # Find the most similar question from the CSV
            solution = find_similar_question(user_issue, df)
            
            # If no good match is found, fall back to OpenAI
            if not solution:
                solution = get_openai_response(user_issue)
            
            # Display the solution
            st.success("Here's a solution for you:")
            st.write(f"**Solution:** {solution}")
    else:
        st.error("Please describe the issue before submitting!")

# Footer with company details
st.markdown("---")
st.markdown("""
    <div style='text-align: center; font-size: small;'>
    © 2024 Exemplify Inc. All rights reserved. | Contact: support@exemplify.com
    </div>
""", unsafe_allow_html=True)

