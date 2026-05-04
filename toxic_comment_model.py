# -----------------------------
# Step 0: Install Libraries
# -----------------------------
!pip install gradio scikit-learn pandas matplotlib wordcloud joblib --quiet

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from wordcloud import WordCloud
import joblib
import gradio as gr

# -----------------------------
# Step 2: Load Dataset
# -----------------------------
df = pd.read_csv("toxic_comment_classification_.csv")
print("First 5 rows:")
print(df.head())
print("Columns:", df.columns)
print("\nLabel distribution:\n", df['label'].value_counts())

# -----------------------------
# Step 3: Preprocess Text and Labels
# -----------------------------
df['comment_text'] = df['comment_text'].astype(str).str.lower().str.replace(r'\n',' ', regex=True)
df['label'] = df['label'].map({'not':0, 'toxic':1})

# -----------------------------
# Step 4: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['label'], test_size=0.2, random_state=42)

# -----------------------------
# Step 5: Vectorize Text
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Step 6: Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Toxic','Toxic']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Toxic','Toxic'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.bar(['Precision','Recall','F1-score'], [precision, recall, f1], color='skyblue')
plt.ylim(0,1)
plt.title("Model Performance Metrics")
plt.show()

# Word Cloud of Toxic Comments
toxic_comments = df[df['label']==1]['comment_text']
text = " ".join(toxic_comments)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Common Words in Toxic Comments")
plt.show()

# -----------------------------
# Step 8: Save Model and Vectorizer
# -----------------------------
joblib.dump(model, "toxic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# -----------------------------
# Step 9: Gradio App
# -----------------------------
def predict_toxic(comment):
    if not comment.strip():
        return "Please enter a comment!"
    vec = vectorizer.transform([comment])
    pred = model.predict(vec)[0]
    return "✅ Not Toxic" if pred==0 else "❌ Toxic"

iface = gr.Interface(
    fn=predict_toxic,
    inputs=gr.Textbox(lines=4, placeholder="Type a comment here..."),
    outputs="text",
    title="Toxic Comment Detection",
    description="Enter a comment to check if it's toxic or not.\n✅ Not Toxic | ❌ Toxic"
)

iface.launch(share=True)
