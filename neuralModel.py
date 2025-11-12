import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
from collections import defaultdict
import torch
import random
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


def formatData(src):
    epdf = pd.read_csv(src, encoding='utf-8', encoding_errors='ignore')
    features = epdf[['original_air_year', 'us_viewers_in_millions', 'season']].dropna(axis=0).to_numpy()
    features = torch.from_numpy(features)
    labels = epdf['imdb_rating'].dropna(axis=0).to_numpy()
    labels = torch.tensor(labels, dtype=torch.long)  
    #labels = epdf.to_string(columns = ['imdb_rating'], header = False, index = False)

    return features, labels

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #create the first hidden layer, takes the embeddings in and outputs the hidden layer sized output
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        #relu layer takes any input and returns a = 0 or >0 value
        self.relu = nn.ReLU()

        #add any other layers here
        #other layers should follow the same linear-activation-linear sandwich strategy
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu3 = nn.ReLU()

        #final hidden layer, takes hidden in outputs our output
        self.output_layer = nn.Linear(self.hidden_size, self.output_size, bias = False)

    def forward(self, x):
        x = self.linear1(x) #[batch_size, sequence_len, hidden_size]
        x = self.relu(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        #now we pool and output our data and pass to our final layer
        res = self.output_layer(x)
        return res



def train(model, train_features, train_labels, epochs, learning_rate):
    batch = 64
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)   

    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    train_history = []
    model.train()

    for j, epoch in enumerate(range(epochs)):
        total_loss = 0
        for i, (features, targets) in enumerate(train_dataloader):
            #reset optimizer
            optimizer.zero_grad()
            #make our predictions
            predictions = model(features)
            #check how the model did and calculate loss
            losses = loss_function(predictions, targets)
            losses.backward()
            #move our optimizer forwards one step
            optimizer.step()
            #tally losses
            total_loss += losses.item()
        if j % 10 == 0:  # Print only every 100 epochs
            avg_loss = total_loss / len(train_dataloader)
            train_history.append(avg_loss)
            print(f"Epoch {epoch}: average training loss: {avg_loss}")
    return train_history


def evaluate(model, test_features, test_labels):
    """
    Evaluate the trained model on test data
    
    Args:
        model: The trained neural network model
        test_features: (tensor)
        test_labels: (tensor)
    
    Returns:
        a dictionary of evaluation metrics (include test accuracy at the minimum)
        (You could import scikit-learn's metrics implementation to calculate other metrics if you want)
    """
    
    ####################### 
    # TODO: Implement the evaluation function
    # Hints: 
    # 1. Use torch.no_grad() for evaluation
    # 2. Use torch.argmax() to get predicted classes
    #######################
    
    model.eval()

    with torch.no_grad():
        prediction = model(test_features)
        predicted_class = torch.argmax(prediction, dim=1)
        #check and add to f1 calculations

    TP = 0
    FP= 0
    TN = 0
    FN = 0
    total = predicted_class.size(dim=0)
    for i in range(predicted_class.size(dim=0)):
        print(f"{i}: prediction: {predicted_class[i]},real value: {test_labels[i]}  ")
        if predicted_class[i] == test_labels[i]:
            if predicted_class[i] >= 7:
                TP+=1
            else:
                TN+=1
        else:
            if predicted_class[i] >=7:
                FP+=1
            else:
                FN+=1



    return {
        'test_accuracy': (TP+TN)/total, 
        'test_precision': TP/(TP+FP),
        'test_recall': TP/(TP+FN),
        'test_f1': (2*(TP/(TP+FP))*(TP/(TP+FN)))/(TP/(TP+FP) + (TP/(TP+FN))),
    }




if __name__ == '__main__':

    train_features, train_labels = formatData("simpsons_episodes.csv")
    val_features, val_labels = formatData("val.csv")
    
    combined_data = list(zip(train_features.tolist(), train_labels.tolist()))
    
    # shuffle
    random.shuffle(combined_data)
    
    # Unzip
    shuffled_features, shuffled_labels = zip(*combined_data)

    # Convert back to a tensor
    shuffled_labels = torch.tensor(shuffled_labels, dtype=torch.long)
    shuffled_features = torch.tensor(shuffled_features)

    # split 80% for train, 20% for test
    split_idx = int(len(shuffled_features) * 0.8) # 80% for train, 20% for test
    
    train_features = shuffled_features[:split_idx]
    train_labels = shuffled_labels[:split_idx]
    
    test_features = shuffled_features[split_idx:]
    test_labels = shuffled_labels[split_idx:]

    print(test_features)
    print(test_labels)
    
    #model time
    input_size = 3
    embedding_dim = 100
    hidden_size = 64
    output_size = 10  # 0-9
    
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Train
    training_history = train(model, train_features, train_labels, epochs=30, learning_rate=0.001)
    
    print(training_history)

    
    # Evaluate
    evaluation_results = evaluate(model, test_features, test_labels)
    
    print(f"Model performance report: \n")
    print(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"Test F1 score: {evaluation_results['test_f1']:.4f}")
    print(f"Test Precision score: {evaluation_results['test_precision']:.4f}")
    print(f"Test Recall score: {evaluation_results['test_recall']:.4f}")

    # Save model weights to file
    outfile = 'models/trained_model.pth'
    torch.save(model.state_dict(), outfile)
    print(f"Trained model saved to {outfile}")

    