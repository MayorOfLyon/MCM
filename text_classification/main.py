import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from dataset import SentimentDataset
from dataset import read_imdb_dataset

# virtual environment: pytorch-nlp
local_model_path = "./pretrained"
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertForSequenceClassification.from_pretrained(local_model_path)

train_df = read_imdb_dataset('aclImdb_v1/train')
test_df = read_imdb_dataset('aclImdb_v1/test')

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-5)

# train
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("length of train_dataloader:", len(train_dataloader))
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # validation
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    avg_val_loss = val_loss / len(val_dataloader)
    accuracy = correct_preds / total_preds
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

# 进行情感分类预测
text = "You are not good at cooking."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)

print("Predicted sentiment:", predictions.item())
