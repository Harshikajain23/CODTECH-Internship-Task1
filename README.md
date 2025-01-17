
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

## Usage

- Navigate to the project directory and run `python app.py`.
- Open your browser and go to `http://127.0.0.1:5000`.
- Type or paste your text into the provided text area.
- Click the **"Analyze Sentiment"** button to see sentiment analysis results.
- Click the **"Summarize Text"** button to generate and display a summary.
- Sentiment analysis and summary results will be displayed below the text area.
- Press `Ctrl + C` in the terminal to stop the application.

## Contributing
  
We appreciate your contributions! Follow these steps to contribute to the project.

- Fork the repository.
- Create a new branch.
- Make and test your changes.
- Commit and push your changes.
- Submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

![screen shot 1](https://github.com/Harshikajain23/CODTECH-Internship-Task1/assets/129208900/903c6f0d-3544-4844-ba35-d4c4076b4a66)

![screen shot 2](https://github.com/Harshikajain23/CODTECH-Internship-Task1/assets/129208900/d9b6189a-d1d9-47f4-b465-ff6a28498001)












