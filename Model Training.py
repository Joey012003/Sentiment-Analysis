import joblib

# Save model
joblib.dump(model, 'sentiment_model.pkl')

# For tokenizer/BERT:
tokenizer.save_pretrained('tokenizer/')
model.save_pretrained('model/')
