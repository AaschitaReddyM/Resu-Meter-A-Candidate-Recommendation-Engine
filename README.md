Resu-Meter: An AI Powered Candidate Recommendation Engine

Resu-Meter is an interactive web application designed to help recruiters find the best candidates for a job position. By leveraging semantic search and generative AI, the tool ranks the candidates based on the relevance of their resumes to a given job description.This streamlines the initial screening process.

🛠️ Tech Stack 
Application Framework: Streamlit
Data Science & NLP:
Embeddings: sentence-transformers
Numerical Operations: NumPy
Keyword Extraction: spaCy
Generative AI: Google Gemini API - gemini-2.5-flash model
Data Handling & Visualization: Pandas, Plotly
File Parsing: PyPDF2, python-docx
Deployment: Docker, Ngrok (for demo)


⬇️ The Installed Libraries list
1.Streamlit: this is the core framework i used to build the interactive web application and all its UI components as well (like buttons, text boxes etc)

2.sentence-transformers: this is a powerful library I used to convert the text from resumes and job description, into meaningful numerical representations called embeddings to further calculate the relevancy and rankings.

3.numpy: a package fundamental for numerical computing in python. I used this to calculate cosine similarity between the embeddings. Crucial for ranking the various candidates as per the relevance to job description.

4.google-generativeai: This is the official google library that lets my application communicate with the Gemini API to generate AI summaries for the candidates.

5.pyngrok: A tool that creates a secure, public URL to my Streamlit application which is running inside a google colab environment to be run on a web browser.

6.PyPDF2, python-docx: These libraries are for file parsing. PyPDF2 is used to extract text from .pdf files and python-docx is used to extract text from .docx files(ms word).

7.spacy: I used this library for adding an extra touch of Natural Language Processing(NLP). I used this because of its robust mechanism to analyse text and extract meaningful keywords.

8.plotly and pandas: It's a powerful duo for data visualization. Pandas is used to organize the final recommendation data into a structured table(a dataframe).Plotly is used to create the interactive bar chart displayed indicating the rankings of the candidates as per the job relevance.



🌟 Key Features

Highlights:
1.This application not only fulfills all the core requirements of the assignment but also includes several advanced features to provide a richer, more insightful user experience. 

2.Upgrading to a next-generation gemini-2.5-flash model provides significant advantages. It delivers superior speed and efficiency, ensuring an instant, responsive user experience. The model's enhanced reasoning and larger context window result in more nuanced and accurate AI summaries, even for complex resumes. This commitment to using a state-of-the-art model ensures the highest quality output and a more reliable application.

Core Requirements: 
Job Description Input: Accepts any job description via a simple text area.

Flexible Resume Input: Supports both bulk file uploads (.pdf, .docx, .txt) and direct text pasting for resumes.

Semantic Ranking:
Generates sophisticated text embeddings for all documents using sentence-transformers.

Computes the cosine similarity between the job description and each resume to quantify relevance.

Top Candidate Display: Clearly lists the top 10 most relevant candidates with their corresponding similarity scores.

Bonus: AI-Powered Summaries:
Integrates with the Google Gemini API to generate a concise, 1-2 sentence summary explaining why each candidate is a great fit for the role.



✨ Above & Beyond Features
NLP Keyword Matching: The engine doesn't just rely on a single similarity score. It uses spaCy to perform Natural Language Processing to extract key skills and nouns from the job description and highlights the keywords matched in each candidate's resume, providing an extra layer of explainability.

Interactive Visualization: The results are prepended with an interactive bar chart created with Plotly. This dashboard-like view gives recruiters an immediate, easy-to-understand visual comparison of the top candidates.

Deployment Ready: The project includes a complete Dockerfile and requirements.txt file. This demonstrates a professional understanding of containerization and reproducibility, ensuring the application can be deployed reliably in any environment.


🔎 Extra Insights about some keywords used 
1. en_core_web_sm: Used in code cell 1. It is the name of the pre-trained model (en = english; core= means the model is a general purpose model that can do many things like part-of-speech identification, tokenization etc; web= the model was primarily trained on the text available on the web including sources from blogs, articles and comments; sm= small version of the bigger model , fast and efficient for keyword extraction task).

2.@st.cache_resource: This is used as a decorator which is a special feature in python. When you place it above a function, it tells streamlit to be smart about that function. Here I'm using it to ensure the large Sentence transformer and spaCy models are loaded into memory only once when the app first starts. This drastically speeds up the application on subsequent runs because it doesn't have to reload these heavy models every time i click on the application.

⚙️How It Works
The application follows a simple yet powerful workflow:

Input: The user provides a job description and a list of candidate resumes (uploaded or pasted).

Text Extraction: The system parses all documents into clean, raw text.

Embedding & Analysis:
The sentence-transformers model converts the job description and each resume into numerical vectors (embeddings).
Simultaneously, spaCy extracts key nouns from the job description.

Ranking: The cosine similarity is calculated between the job description vector and each resume vector.

Output Generation: The candidates are ranked by their similarity score. For each top candidate, the system generates an AI summary and identifies matched keywords.

Display: The final ranked list, interactive chart, and detailed results are presented to the user in the Streamlit interface.


📝 Explanation about Helper Functions used 
These are the specialised functions that are defined to perform the core logic of the application.

1.extract_text_from_file(): It reads uploaded files. Checks the file type(.pdf, .docx, .txt) and uses the respective library(PyPDF2 or python-docx) to open the file and extract all the text.

2.cosine_similarity(): It takes in 2 numerical vectors(the embeddings from job description and resumes) and calculates the cosine similarity score which is a number between 0 and 1 that represents how similar the embeddings are. The more the score is, the better fit the candidate for the job.

3.get_gemini_summary(): It handles all the communication with the Gemini API. It defines a detailed and helpful prompt, sends it to the gemini model and returns the AI generated summary.

4.extract_keywords(): This is an NLP function. It uses the pre-loaded spaCy model to process text, identify all the nouns and proper nouns, and then filter them to return a clean list of meaningful keywords and eliminate special characters.


Project Summary: Resu-Meter
The "Resu-Meter" project successfully delivered a sophisticated AI Candidate Recommendation Engine. The application is fully functional and performs flawlessly in a local environment, proving its robust design and meeting all core objectives.

Local Success: The app was completely built and validated, perfectly handling resume parsing, AI-based relevance ranking, and generative candidate summaries.

Deployment Status: The application's code is stable and ready for deployment. Due to time constraints, the final step of adapting the app to the specific resource limitations of a free-tier cloud service was identified as the next step on the project roadmap.

In conclusion, "Resu-Meter" is a proven success, demonstrating the ability to build and deliver a complex AI application from concept to a fully-realized product.

Screenshots of outputs:











🚀 Setup and Local Installation
To run this project on your local machine, please follow these steps:
Clone the Repository
Bash
git clone https://github.com/your-username/resu-meter.git
cd resu-meter


Create a Virtual Environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install Dependencies
Bash
pip install -r requirements.txt


Set Up Environment Variables Create a file named .env in the root directory and add your API keys:
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
NGROK_AUTH_TOKEN="YOUR_NGROK_AUTH_TOKEN" 
The application code will need to be slightly modified to load variables from a .env file (e.g., using python-dotenv). The current version in the notebook uses Colab Secrets.
Run the Streamlit App
Bash
streamlit run app.py


