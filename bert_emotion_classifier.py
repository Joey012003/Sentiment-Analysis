import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load dataset
df = pd.read_csv('twitter_training.csv', header=None)
df.columns = ['tweet_id', 'entity', 'label', 'text']
df = df[df['label'].isin(['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust'])]

# Map labels to integers
label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
id2label = {idx: label for label, idx in label2id.items()}
df['label_id'] = df['label'].map(label2id)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Hugging Face Dataset
train_dataset = Dataset.from_dict({**train_encodings, 'label': train_labels})
val_dataset = Dataset.from_dict({**val_encodings, 'label': val_labels})

# Model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Train and Evaluate
trainer.train()
preds = trainer.predict(val_dataset)
pred_labels = torch.argmax(torch.tensor(preds.predictions), axis=1)

# Evaluation
print(classification_report(val_labels, pred_labels, target_names=label2id.keys()))
