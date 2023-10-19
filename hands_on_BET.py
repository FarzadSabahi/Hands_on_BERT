import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.nn import CrossEntropyLoss

# Load a pre-trained model and tokenizer from the 'transformers' library
teacher = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create your smaller student model. Here, for simplicity, I’m  using the same architecture for the student.
# In a real-world scenario, 'student' should be a smaller, more efficient model.
student = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Assume you have a function 'create_dataset' that prepares your dataset.
# Your dataset should be tokenized and formatted correctly.
def create_dataset():
    texts = ["example sentence 1", "example sentence 2", "example sentence 3"]
    labels = [0, 1, 0]  # Dummy labels
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
    return dataset

dataset = create_dataset()
data_loader = DataLoader(dataset, batch_size=1)

# Use a loss function and optimizer for training
loss_function = CrossEntropyLoss()
optimizer = AdamW(student.parameters(), lr=1e-5)

# Knowledge distillation usually involves a temperature term to smooth the probability distribution
temperature = 1.5

# Training loop
student.train()
for epoch in range(3):  # For simplicity, I’m  using 3 epochs. You'd want more epochs typically.
    total_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()

        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            teacher_logits = teacher(input_ids, attention_mask=attention_mask).logits

        # The student does a forward pass and we calculate the loss directly between student and teacher logits
        student_logits = student(input_ids, attention_mask=attention_mask).logits
        loss = loss_function(student_logits/temperature, torch.argmax(teacher_logits, dim=1))
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

# Save your trained student model
student.save_pretrained("./optimized_model")
