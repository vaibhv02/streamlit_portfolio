import streamlit as st
import requests
from PIL import Image
import base64

# Set page configuration
st.set_page_config(page_title="Vaibhav Sharma's Portfolio", page_icon=":briefcase:", layout="centered")

# Function to fetch a single GitHub repository's details
def fetch_github_project(username, repo_name):
    api_url = f"https://api.github.com/repos/{username}/{repo_name}"
    response = requests.get(api_url)
    if response.status_code == 200:
        repo = response.json()
        project_info = {
            "name": repo['name'],
            "description": repo['description'] if repo['description'] else "No description provided.",
            "url": repo['html_url']
        }
        return project_info
    else:
        st.error(f"Error fetching repository: {repo_name}")
        return None

# Set GitHub username and selected repos (replace these with actual GitHub repos)
github_username = "vaibhv02"  # Replace with your GitHub username
selected_repos = ["GenAI_Project","Object_Detection_in_Real_Time_Video_Stream", "twitter_sentiment_analysis", "Customer_Segmentation", "House_Price_Prediction"]  # Replace with your repo names

# Load your image
image = Image.open("Image.png")  # Replace with your photo file

# App layout
st.title("Vaibhav Sharma")
st.write("**Data Science/Machine Learning Specialist**")

# Display your image
st.image(image, caption="Vaibhav Sharma", width=150)

# Specify the path to your resume file
resume_path = "Vaibhav Sharma Resume.pdf"

# Create a download button using Streamlit's built-in function
with open(resume_path, "rb") as file:
    st.download_button(
        label="Download Resume",
        data=file,
        file_name="Vaibhav Sharma Resume.pdf",
        mime="application/pdf",
        key="download_button"
    )

# Personal details
st.write(""" 
**Email**: vaibhavshrma002@gmail.com  """
"""
**GitHub**: https://github.com/vaibhv02   """
"""
**LinkedIn**: https://www.linkedin.com/in/er-vaibhav-sharma
""")

# Summary
st.subheader("Summary")
st.info("""
- Seasoned Python Developer with a strong focus on data analysis, machine learning, and software development. Demonstrated experience in building scalable solutions, optimizing models, and utilizing advanced algorithms. Proficient in applying machine learning techniques to address real-world challenges, including sentiment analysis, customer segmentation, and predictive analytics.

- Hands-on experience in real-time object detection and deep learning frameworks such as YOLOv3 and TensorFlow. A passion for solving complex problems, with a commitment to continuous learning of new technologies and contributing to innovative projects. Ready to leverage skills in data science and AI to deliver impactful solutions in a professional environment.
""")

# Qualifications
st.subheader("Qualifications")
st.write("""
**Bacholers of Technology - Computer Science Engineering** (2021-2025)

- Rayat Bahra University, Punjab 


**College Courses:**  Data Structures, Algorithms, Operating Systems,Computer Networks, Database Management Systems, Software Engineering

##### Certifications & Self-Learning
- **IBM Data Science Professional Certificate**: Completed modules in Python, Data Science, and Machine Learning, covering data analysis, statistical techniques, and model evaluation.
- **Machine Learning and Deep Learning - NPTEL**: Gained in-depth knowledge of supervised/unsupervised learning, neural networks, and real-world applications.
- **Full Stack Development Program - Excellence Education**: Learned JavaScript, React, and backend development with Node.js and databases like MySQL.
- **Full Stack Web Development - MCP Technology**: Focused on both front-end and back-end web technologies, improving expertise in scalable web applications.
""")

# Projects
st.header("Projects")

# Project 1: GenAI Project
st.subheader("GenAI Project")
st.write("[GitHub](https://github.com/vaibhv02/GenAI_Project)")
st.write("""
**Objective:** Implement a QA Bot using a Retrieval-Augmented Generation (RAG) model.

**Technologies Used:** Python, RAG model, Vector Database.

**Description:**
Developed a QA Bot that combines retrieval and generative models for efficient document retrieval and coherent answer generation.

**Process:**
- Designed the QA Bot architecture.
- Integrated a vector database for fast document retrieval.
- Implemented the RAG model for accurate answers.
         
**Outcome:**
- `85%` accuracy in answering queries.
- Reduced response time.

""")



# Project 2: Real-Time Object Detection in Video Streams
st.subheader("Real-Time Object Detection in Video Streams")
st.write("[GitHub](https://github.com/vaibhv02/Object_Detection_in_Real_Time_Video_Stream)")
st.write("""
**Objective:** Build a real-time object detection system using YOLOv3 and OpenCV.

**Role:** Project Lead, responsible for developing and optimizing the detection pipeline.

**Technologies Used:** Python, YOLOv3, OpenCV, NumPy.

**Process:**
- Integrated YOLOv3 for detecting `40+` object categories in video streams with high accuracy.
- Optimized video processing to achieve `30 FPS` and reduce latency by `45%`.
- Applied techniques for reducing false positives and enhancing detection precision.

**Outcome:**
- Achieved `90%+` detection accuracy, delivering precise object recognition in live video feeds.
- Reduced processing latency, ensuring smooth real-time performance.
""")


# Project 3: Twitter Sentiment Analysis
st.subheader("Twitter Sentiment Analysis")
st.write("[GitHub](https://github.com/vaibhv02/twitter_sentiment_analysis)")
st.write("""
**Objective:** Develop a machine learning model to classify sentiments (positive, negative, neutral) from `1.6 million` tweets.

**Role:** Lead Developer, responsible for data cleaning, feature engineering, and model optimization.

**Technologies Used:** Python, Scikit-learn, TF-IDF, Logistic Regression.

**Process:**
- Preprocessed raw tweet data using tokenization, stopword removal, and vectorization with TF-IDF.
- Performed exploratory data analysis (EDA) to identify trends and patterns in the tweet data.
- Built and fine-tuned a logistic regression model, achieving `84%` accuracy.

**Outcome:**
- Improved data processing speed by `30%`, reducing overall model runtime.
- The model was able to classify tweets with high accuracy, providing valuable insights into public sentiment.
""")


# Project 4: Customer Segmentation and Predictive Analytics
st.subheader("Customer Segmentation and Predictive Analytics")
st.write("[GitHub](https://github.com/vaibhv02/Customer_Segmentation)")
st.write("""
**Objective:** Segment customers for targeted marketing campaigns and predict customer behavior.

**Role:** Data Scientist, focusing on clustering techniques and predictive analytics.

**Technologies Used:** Python, Pandas, NumPy, K-Means Clustering, Scikit-learn.

**Process:**
- Segmented `1,000+ customers` using K-Means clustering, optimizing marketing campaigns.
- Performed feature engineering on over `500,000 rows` of customer data to improve model efficiency.
- Developed models to predict customer lifetime value and retention.

**Outcome:**
- Improved customer retention by `45%`, contributing to a `15%` increase in marketing campaign effectiveness.
- Provided actionable insights to the marketing team, enhancing targeted marketing efforts by `87%`.
""")

# Skills
# Technical Skills
st.header("Technical Skills")
st.write("""
- **Data Analysis:** `Pandas`, `NumPy`, `Feature Engineering`, `Model Evaluation`
- **Machine Learning:** `Regression`, `K-Means`, `Random Forest`, `Deep Learning`
- **Programming Languages:** `Python`, `Java`
- **Database Management:** `MySQL`, `MongoDB`
- **Tools:** `Jupyter Notebook`, `Git`
- **Frameworks:** `OpenCV`, `Scikit-Learn`, `YOLOv3`
""")

