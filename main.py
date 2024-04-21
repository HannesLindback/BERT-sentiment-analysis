import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import transforms
from preprocessing import Data, Tokenize, Process
from model import SentimentAnalysis
from transformers import logging
from transformers import AdamW
from tqdm import tqdm
import numpy as np


class MovieReviewClassifier:

    def __init__(self, model=None, n_epochs=2, batch_size=32) -> None:
        """Implements a text classification model for classifying movie
        reviews as positive or negative.
        
        Uses a pre-trained BERT model with a one-layer MLP for the classification.
        
        Fine-tunes the BERT model's final layer and freezes the rest of the layers.
        
        Will classify a review as either positive or negative."""
        
        self.model = SentimentAnalysis() if model is None else model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss_function = nn.BCELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        self.training_loss_ = list()                                  
        self.training_accuracy_ = list()

    def fit(self, X_raw, y):
        """Fits model to the given data."""
        
        print('Fitting model...')

        dataset = Data(X=X_raw, y=y, transform=transforms.Compose([Tokenize()]))
        
        self.model.train()
        
        for epoch in range(self.n_epochs):
            
            with tqdm(DataLoader(dataset, batch_size=self.batch_size, shuffle=True), 
                    total=len(dataset.X)//self.batch_size+1, unit="batch", 
                    desc="Epoch %i" % epoch) as batches:
                
                for batch in batches:
                    loss, accuracy = self._fit(batch)
                    
                    batches.set_postfix(loss=loss, accuracy=accuracy)
                    
            print('Mean accuracy for epoch: ', sum(self.training_accuracy_) / len(self.training_accuracy_))
            print('Accuracies: ', self.training_accuracy_[:10] + self.training_accuracy_[-10:])

    def _fit(self, batch):
        """Performs the forward and backward passes for one epoch."""
        
        inputs, targets = batch['X'], batch['y'].cuda().to(dtype=torch.float32)

        # Freeze the parameters of the embedding layers and all encoder layers except the last.
        modules = [self.model.model.embeddings, *self.model.model.encoder.layer[:-1]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        # Forward
        self.model.zero_grad()
        output = self.model(inputs)
        
        output = output.flatten()
        
        # Backward
        loss = self.loss_function(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()

        predictions = torch.where(output>0.5, 1.0, 0.0).to(dtype=torch.float32)
        correct = (predictions == targets).sum().item()
        accuracy = (correct / self.batch_size) * 100

        self.training_accuracy_.append(accuracy)
        self.training_loss_.append(loss.item())

        return loss.item(), accuracy

    def score(self, X_raw, y):
        """Scores the model on the given test dataset."""
        
        print('Scoring model...')

        dataset = Data(X=X_raw, y=y, transform=transforms.Compose([Tokenize()]))

        self.model.eval()
        with torch.no_grad(): 
            n_correct = 0
            n_total = 0

            for i, batch in enumerate(DataLoader(dataset, batch_size=self.batch_size, shuffle=False)):
                inputs, targets = batch['X'], batch['y'].cuda().to(dtype=torch.float32)
                output = self.model(inputs)
                output = output.flatten()
                predictions = torch.where(output>0.5, 1.0, 0.0).to(dtype=torch.float32)
                n_correct += (predictions == targets).sum().item()
                n_total += self.batch_size

                if i == 100:
                    print(f'Processed {i} out of {len(dataset)} batches')
                    break

        return n_correct/n_total

    def predict(self, X_raw):
        """Predicts the label for inputs."""
        
        y = [-1, -1] # Dummy label
        dataset = Data(X=X_raw, y=y, transform=transforms.Compose([Tokenize()]))
        for batch in DataLoader(dataset, batch_size=2, shuffle=False):
            inputs = batch['X']
            output = self.model(inputs)
            output = output.flatten()
            prediction = torch.where(output>0.5, 1.0, 0.0).to(dtype=torch.float32)[0].item()
        return int(prediction)

    def transform(self, path):
        """Transform data to correct format."""
        
        process = Process()
        return process(path)
    

def plot_loss(model, X_train, batch_size):
    """Function for plotting the loss and accuracy during training."""

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    ax.set_title("Plot for loss over epochs")
    ax.plot(model.training_loss_, 'b-')
    ax.set_ylabel("Training Loss", color='b')
    ax.set_xlabel("Epoch")
    ax.tick_params(axis='y', labelcolor='b')
    ax = ax.twinx()
    ax.plot(model.training_accuracy_, 'r-')
    ax.set_ylabel("Accuracy [%]", color='r')
    ax.tick_params(axis='y', labelcolor='r')
    a = list(ax.axis())
    a[2] = 0
    a[3] = 100
    ax.axis(a)
    t = np.arange(0, len(model.training_accuracy_), len(X_train)//batch_size+1)
    ax.set_xticks(ticks=t)
    ax.set_xticklabels(labels=np.arange(len(t)))
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    logging.set_verbosity_error()
    
    model = SentimentAnalysis()
    classifier = MovieReviewClassifier(n_epochs=1, batch_size=32)
    texts, labels = classifier.transform(path='aclImdb/train')
    classifier.fit(texts, labels)

    plot_loss(classifier, texts, classifier.batch_size)

    texts, labels = classifier.transform(path='aclImdb/test')
    score = classifier.score(texts, labels)
    print(score)
    print(classifier.predict("My boyfriend and I went to watch The Guardian.At first I didn't want to watch it, but I loved the movie- It was definitely the best movie I have seen in sometime.They portrayed the USCG very well, it really showed me what they do and I think they should really be appreciated more.Not only did it teach but it was a really good movie. The movie shows what the really do and how hard the job is.I think being a USCG would be challenging and very scary. It was a great movie all around. I would suggest this movie for anyone to see.The ending broke my heart but I know why he did it. The storyline was great I give it 2 thumbs up. I cried it was very emotional, I would give it a 20 if I could!"))
    
