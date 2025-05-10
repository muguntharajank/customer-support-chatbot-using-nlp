# customer-support-chatbot-using-nlp


🤖 Chatbot AI using E-Commerce Data
      This project builds a basic AI-powered chatbot trained on real-world e-commerce order data. The chatbot can answer user queries related to orders, shipping, payments, and products using intent classification and machine learning.

📌 Overview
   The goal of this project is to:

         Preprocess and merge datasets from an e-commerce platform.

         Clean and prepare queries from structured fields like product categories.

         Train an intent classification model using either Logistic Regression or a simple Deep Learning model.

         Enable the chatbot to respond to user inputs based on learned intents.

📁 Dataset Used
         The chatbot is trained on merged data from the Olist E-Commerce Dataset:

         olist_orders_dataset.csv – Customer order metadata.

         olist_order_items_dataset.csv – Products within each order.

         olist_order_payments_dataset.csv – Payment details.

         olist_products_dataset.csv – Product metadata (including category names).

      These are merged to simulate question-answer pairs and chatbot queries.

🔧 Installation
      Python Version
      Python 3.7+

Install dependencies:
      bash
      Copy
      Edit
      pip install pandas scikit-learn tensorflow


📂 File Structure
      text
      Copy
      Edit
      chatbot_ai.ipynb             <- Main notebook
olist_order_dataset/

│
├── olist_orders_dataset.csv

├── olist_order_items_dataset.csv

├── olist_order_payments_dataset.csv

└── olist_products_dataset.csv


🧠 Workflow

1. Library Import
      Essential ML and deep learning libraries like Scikit-learn, TensorFlow, Pandas.

2. Dataset Loading
      Load multiple CSV files and preview them to ensure correctness.

3. Merging & Cleaning
      Join the datasets based on order_id and product_id.

      Clean product categories or text fields using .lower() and .strip().

4. Text Vectorization
      Use TfidfVectorizer to transform cleaned text into numerical features.

5. Label Encoding
      Use LabelEncoder to encode categorical labels for training.

6. Model Training

   
      Option A: Train a LogisticRegression model from Scikit-learn.

      Option B: Train a simple TensorFlow Sequential model with Dense layers.

7. Prediction & Accuracy:
   
      Evaluate the model using accuracy_score.

      Check performance on test set.

8. Chatbot Function
      Take user input, transform it using TF-IDF.

      Predict intent and respond accordingly.



💬 Sample Code Snippet
      python
      Copy
      Edit
def get_response(user_input):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)
    return label_encoder.inverse_transform(pred)[0]


✅ Model Evaluation

   Accuracy printed after training.

   Supports both traditional ML and neural network classifiers.

   Use either train_test_split or manual test queries.


📌 Notes:  

   The chatbot logic is built primarily for educational and demo purposes.

   Performance depends heavily on how you structure and label training queries.

🔄 Environment Compatibility
✅ Jupyter Notebooks

✅ Google Colab (includes Drive mounting instructions)
