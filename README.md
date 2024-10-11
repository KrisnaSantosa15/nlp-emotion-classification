# 🎭 Emotion Classification from Comments

## 🚀 What project is this?

Welcome to my Emotion Classification project! The goal of this project is to classify emotions in text data using Natural Language Processing (NLP) techniques. The dataset used in this project is the [Emotion dataset](https://www.kaggle.com/datasets/abdallahwagih/emotion-dataset/data) from Kaggle.

Content: <br>
Format: CSV <br>
Labels: ['anger', 'joy', 'fear']

Use Cases:
- Sentiment analysis
- Emotion classification
- Emotion-aware applications
- Customer feedback analysis
- Social media sentiment monitoring
- Chatbot and virtual assistant training

The dataset consist of 5.934 samples with 3 labels: anger, joy, and fear.

### 🌟 Key Features

- **Preprocessing**: Utilizes NLTK for stopword removal and intelligent text cleaning.
- **Deep Learning**: Employs a state-of-the-art LSTM neural network for nuanced emotion detection.
- **Architecture**: Implements dropout layers to prevent overfitting and ensure model generalization.
- **High Accuracy**: Achieves an impressive 93.10% validation accuracy.

## 🛠️ Technical Stack

- **Python**: The backbone of this implementation.
- **Pandas**: For efficient data manipulation and analysis.
- **TensorFlow & Keras**: For building and training the emotion classification model.
- **NLTK**: Enabling advanced natural language processing techniques.
- **Scikit-learn**: Facilitating data splitting and preprocessing.

## 📊 Model Architecture

This emotion classification model use architecture with the following layers:

1. **Embedding Layer**: Transforms words into dense vectors of fixed size.
2. **LSTM Layer**: Captures long-term dependencies in the text.
3. **Dense Layers**: Provide deep feature extraction and classification.
4. **Dropout Layers**: Prevent overfitting.

## 🔍 Data Insights

The project use a dataset (`Emotion_classify_Data.csv`) containing comments and their associated emotions. This project focus on classifying three primary emotions: fear, anger, and joy.

## 🚀 Getting Started

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script to train and evaluate the model.

## 📈 Results and Performance

This model achieves:
- Training Accuracy: 99.92%
- Validation Accuracy: 93.10%

These results underscore the model's ability to generalize well to unseen data, striking a balance between fitting the training data and avoiding overfitting.

## 🎯 Future Enhancements

- Experiment with different architectures (e.g., transformers) for potential performance improvements.
- Use other dataset with more labels for multi-class emotion classification.
- Develop a user-friendly web interface for real-time emotion classification.

## 🤝 Contribute

Feel free to fork this repository, open issues, or submit PRs. Any contributions you make are greatly appreciated!

## 📚 Learn More

For a deep dive into the code and methodology, check out my Jupyter notebook in this repository.

---

🌟 Star this repository if you find it interesting!

Best regards, <br>
Krisna Santosa