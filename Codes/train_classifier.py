# train_classifier.py
"""
Train a scam/legit classifier using DistilBERT embeddings.
Input: CSV with 'transcript' and 'label' (scam/legit)
Output: classifier.joblib + embedder_model.txt
"""

import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import joblib
import os
from tqdm import tqdm

def get_distilbert_embeddings(texts, tokenizer, model, device):
    """Generate embeddings by taking [CLS] token output"""
    embeddings = []
    for txt in tqdm(texts, desc="Embedding"):
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:,0,:].cpu().numpy()
        embeddings.append(cls_embedding[0])
    return embeddings

def main(args):
    df = pd.read_csv(args.csv)
    assert 'transcript' in df.columns and 'label' in df.columns, "CSV must have 'transcript' and 'label'"

    df['y'] = df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'scam' else 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

    print("Embedding transcripts with DistilBERT...")
    X = get_distilbert_embeddings(df['transcript'].tolist(), tokenizer, model, device)
    y = df['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=['legit','scam']))

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.out_dir, "classifier.joblib"))

    with open(os.path.join(args.out_dir, "embedder_model.txt"), "w") as f:
        f.write("distilbert-base-uncased")

    print("Saved classifier + embedder info to", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Labeled transcript CSV")
    parser.add_argument("--out_dir", default="models", help="Output directory")
    args = parser.parse_args()
    main(args)
