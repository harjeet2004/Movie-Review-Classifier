🎬 Movie Review Classifier

This repository contains an AI-powered web application built with TensorFlow/Keras and Streamlit, which classifies IMDB movie reviews as Positive or Negative.

The model is trained on the IMDB dataset and deployed on Streamlit Cloud for easy access.

🔗 Live App

👉 Movie-Review-Classifier

📝 Project Description
🎬 Sentiment Analysis on IMDB Reviews

Input: Any movie review text

Output:

Sentiment → Positive / Negative

Prediction Score → Probability value from the trained model

This project demonstrates the use of Recurrent Neural Networks (SimpleRNN) to handle sequential data (sentences) for text classification tasks.

🛠 Technologies Used

Python 3.10+

TensorFlow / Keras

Streamlit

NumPy & Pandas

Scikit-learn

Git & GitHub

Streamlit Cloud

📁 Project Structure
.
├── SimpleRNN/
│   ├── app.py                 # Streamlit app for movie review sentiment classification
│   ├── simple_rnn_imdb.h5     # Trained RNN model (IMDB dataset)
│   ├── prediction.ipynb       # Notebook for testing predictions
│   └── proj.ipynb             # Notebook for model training
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md                  # This file

💻 How to Run Locally
1. Clone the Repository
git clone https://github.com/harjeet2004/Movie-Review-Classifier.git
cd Movie-Review-Classifier

2. Create & Activate a Virtual Environment
conda create -n rnn-env python=3.10
conda activate rnn-env

3. Install Required Libraries
pip install -r requirements.txt

4. Run the App
streamlit run SimpleRNN/app.py


Visit:
http://localhost:8501

🚀 How to Deploy on Streamlit Cloud

Push your code to GitHub

Go to Streamlit Cloud

Click New app and choose your repo

Set the main file → SimpleRNN/app.py

Add requirements.txt

Click Deploy

🧠 What I Learned

Handling sequential text data with RNNs

Using embedding layers and SimpleRNN for natural language tasks

Converting words into integer indices and padded sequences for model input

Building clean UI apps with Streamlit

Debugging deployment issues with model paths & .gitignore rules

Publishing ML apps for public access via Streamlit Cloud

🙋‍♂️ Author

Built with ❤️ by Harjeet Singh Pannu — a college student exploring AI, deep learning, and real-world ML deployment.

📜 License

This project is licensed under the MIT License — feel free to fork, extend, and contribute!
