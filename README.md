# 📧 Spam Email Detection

A machine learning web application that classifies email messages as **Spam** or **Not Spam** using a Naive Bayes classifier, built with Python and Streamlit.

---

## 🚀 Demo

Enter any email message into the app and click **Check** to instantly find out whether it's spam.

---

## 🧠 How It Works

1. **Data Loading** — Reads a labeled dataset (`spam.csv`) containing email messages and their categories (`ham` / `spam`).
2. **Preprocessing** — Removes duplicate entries and maps labels to human-readable values (`Not Spam` / `Spam`).
3. **Feature Extraction** — Uses `CountVectorizer` with English stop-word removal to convert raw text into numerical feature vectors.
4. **Model Training** — Trains a `MultinomialNB` (Multinomial Naive Bayes) classifier on 80% of the data.
5. **Prediction** — Accepts user input via a Streamlit UI and returns a real-time spam classification.

---

## 🗂️ Project Structure

```
spam-email-detection/
├── backend.py      # Main application — model training, prediction, and Streamlit UI
└── spam.csv        # Dataset with labeled email messages
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas | Data loading and preprocessing |
| scikit-learn | Feature extraction (`CountVectorizer`) and model (`MultinomialNB`) |
| Streamlit | Interactive web interface |

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/SameenYounas1/spam-email-detection.git
cd spam-email-detection
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn streamlit
```

### 3. Update the dataset path

In `backend.py`, update the path to `spam.csv` to match your local setup:

```python
# Change this line:
data = pd.read_csv("C:\\Users\\12c\\Documents\\sameen\\spam email detector\\spam.csv")

# To a relative path:
data = pd.read_csv("spam.csv")
```

### 4. Run the app

```bash
streamlit run backend.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📊 Dataset

The `spam.csv` file contains two columns:

| Column | Description |
|---|---|
| `Category` | Label — `ham` (not spam) or `spam` |
| `Message` | The raw email/SMS text |

---

## 🖥️ Usage

1. Launch the app with `streamlit run backend.py`.
2. Type or paste an email message into the text area.
3. Click the **Check** button.
4. The app displays whether the message is **Spam** or **Not Spam**.

---

## 📌 Known Issues / Improvements

- The dataset path in `backend.py` is currently hardcoded to an absolute local path — update it to `spam.csv` for portability.
- The model is retrained from scratch on every app launch; consider saving/loading the trained model with `joblib` for faster startup.
- Model accuracy can be improved by experimenting with TF-IDF vectorization or other classifiers (e.g., Logistic Regression, SVM).

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
