
# Sentiment Analyser and Text Summariser Application

Name: Harshika Jain   
Company: CODTECH IT SOLUTIONS  
ID: CT6AIO7  
Domain: Artificial Intelligence  
Duration: 20 June 2024 to 20 August 2024  

## Overview

This project is a web application designed to analyze the sentiment of user-provided text and generate concise summaries. Utilizing a combination of Artificial Intelligence (AI) techniques and Natural Language Processing (NLP), the application offers insights into the emotional tone of the text and provides a brief summary of the main content.

## Features
#### Sentiment Analysis:  
- **TextBlob:** Determines if text is positive, negative, or neutral.  
- **VADER:** Provides detailed sentiment scores and highlights the dominant sentiment with its percentage.  
- **Naive Bayes Classifier:** Additional sentiment analysis using a trained classifier (currently commented out).

#### Text Summarization:  
- **Summarization:** Generates concise summaries by extracting key sentences based on word frequency.  

## Technology Used  
- **Python 3.x** (Core language for backend and data processing)
- **Flask** (Web framework for building web applications)
- **NLTK** (Natural Language Toolkit for text processing and sentiment analysis)
- **TextBlob** (Simplifies text processing and sentiment analysis)
- **TensorFlow** (Machine learning framework for advanced tasks)
- **HTML/CSS/JavaScript** (Frontend development for user interface and interactivity)

## Setup and Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)
- Virtual environment (optional but recommended)

### Step 1: Clone the Repository

``` 
git clone https://github.com/Harshikajain23/CODTECH-Internship-Task1.git
cd CODTECH-Internship-Task1
```

### Step 2: Set Up Virtual Environment (Optional)

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt 
```

### Step 4: Download NLTK Data

In your Python environment, run the following script to download the necessary NLTK data files:

```
import nltk
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

### Step 5: Run the Application

```
python app.py
```
This will start the Flask development server. By default, it runs on http://127.0.0.1:5000.  
Now you can access the project by navigating to http://127.0.0.1:5000 in your web browser.







