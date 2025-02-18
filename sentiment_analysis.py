import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def get_device():
    """Get the appropriate device (CUDA if available, otherwise CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset_and_preprocess():
    """Load and preprocess dataset with efficient tokenization"""
    from datasets import load_dataset

    # Load smaller subset for training
    dataset = load_dataset("imdb", split="train[:20%]+test[:10%]")
    dataset = dataset.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=16)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return dataset["train"], dataset["test"]


def initialize_models(device):
    """Initialize teacher and student models on the appropriate device"""
    teacher = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ).to(device)

    student = AutoModelForSequenceClassification.from_pretrained(
        "prajjwal1/bert-mini", num_labels=2
    ).to(device)

    # Disable teacher gradients for memory efficiency
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher.eval(), student.train()


class DistillationTraining:
    def __init__(self, student, teacher, train_data, val_data, device):
        self.student = student
        self.teacher = teacher
        self.train_data = train_data
        self.val_data = val_data
        self.device = device

        # Adjust batch size based on device
        self.batch_size = 32 if torch.cuda.is_available() else 8
        self.num_epochs = 3
        self.temperature = 2.0
        self.alpha = 0.5

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(student.parameters(), lr=5e-5, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_data) * self.num_epochs // self.batch_size,
        )
        self.loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, student_outputs, teacher_outputs, labels):
        """Hybrid loss combining distillation and task loss"""
        # Distillation loss
        soft_teacher = torch.nn.functional.softmax(
            teacher_outputs.logits / self.temperature, dim=-1
        )
        soft_student = torch.nn.functional.log_softmax(
            student_outputs.logits / self.temperature, dim=-1
        )
        loss_distill = self.loss_fn(soft_student, soft_teacher) * (self.temperature**2)

        # Student task loss
        loss_task = torch.nn.functional.cross_entropy(student_outputs.logits, labels)

        return self.alpha * loss_distill + (1.0 - self.alpha) * loss_task

    def train_epoch(self, dataloader):
        """Training loop with automatic device placement"""
        self.student.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()

            # Forward passes
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
            student_outputs = self.student(**inputs)

            # Compute and backpropagate loss
            loss = self.compute_loss(student_outputs, teacher_outputs, labels)
            loss.backward()

            # Update parameters
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Clear cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Validation with automatic device placement"""
        self.student.eval()
        total_acc = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k != "label"
                }
                labels = batch["label"].to(self.device)

                outputs = self.student(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                total_acc += (preds == labels).sum().item()

                # Clear cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return total_acc / len(dataloader.dataset)

    def run_training(self):
        """Complete training process with device-aware checkpointing"""
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),  # Enable pin_memory for CUDA
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
        )

        print(f"Training on device: {self.device}")

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)
            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s"
            )

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

        self.save_final_model()

    def save_checkpoint(self, epoch):
        """Save intermediate checkpoints"""
        checkpoint_path = f"bert-mini-sentiment-epoch{epoch}.bin"
        torch.save(self.student.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_final_model(self):
        """Save final model with Hugging Face format"""
        output_dir = "final_sentiment_model"
        os.makedirs(output_dir, exist_ok=True)

        # Move model to CPU before saving
        self.student.cpu().save_pretrained(output_dir)
        AutoTokenizer.from_pretrained("prajjwal1/bert-mini").save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


def predict_sentiment(review_text, model_path="final_sentiment_model"):
    """
    Predict sentiment for a given review text.

    Args:
        review_text (str): The text review to analyze
        model_path (str): Path to the saved model directory

    Returns:
        dict: Dictionary containing the prediction ('positive' or 'negative') and confidence score
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        review_text, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][prediction[0]].item()

    # Convert prediction to label
    sentiment = "positive" if prediction.item() == 1 else "negative"

    # Clear GPU memory if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            "negative": probabilities[0][0].item(),
            "positive": probabilities[0][1].item(),
        },
    }


def main():
    '''# Get the appropriate device
    device = get_device()
    print(f"Using device: {device}")

    # Load and prepare data
    train_data, val_data = load_dataset_and_preprocess()

    # Initialize models on the appropriate device
    teacher, student = initialize_models(device)

    # Create and run the distiller
    distiller = DistillationTraining(student, teacher, train_data, val_data, device)
    distiller.run_training()'''

    # review = "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout."
    # review = "I kind of liked the movie. It was not how I expected it to be but it was good enough to watch for me."
    # review = "I did not like this movie at all. The acting was terrible and the storyline was confusing."
    # review = "While the actors were good, the storyline was not on par with the hype surrounding the movie."
    # review = "Most people loved the movie. I did not like it at all."
    # result = predict_sentiment(review, model_path="final_sentiment_model")

    # print("\nReview:", review)
    # print(f"\nSentiment: {result['sentiment'].upper()}")
    # print(f"Confidence: {result['confidence']:.2%}")
    # print("\nProbabilities:")
    # print(f"Positive: {result['probabilities']['positive']:.2%}")
    # print(f"Negative: {result['probabilities']['negative']:.2%}")


if __name__ == "__main__":
    main()
