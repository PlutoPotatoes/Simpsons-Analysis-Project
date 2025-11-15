import csv
import pandas as pd 
from tabulate import tabulate
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

lines = "simpsons_script_lines.csv"
episodes = "simpsons_episodes.csv"
episodesDf = pd.read_csv(episodes)
linesDf = pd.read_csv(lines)
episodesDf["good_or_bad"] = 0

for index, row in episodesDf.iterrows():
    if(row["imdb_rating"] > 7.3491):
        episodesDf.loc[index, "good_or_bad"] = 1

#drop nan values in episode df
episodesDf = episodesDf.dropna()

#episodesDf = episodesDf.astype("float64", errors='ignore')

#print(episodesDf.head(20))
#print(episodesDf.iloc[409])

class SimpleNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleNN, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    # Define the NN layers
    self.linear1 = nn.Linear(self.input_size, self.hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(self.hidden_size, self.output_size, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # Moving from one layer to the next
    linear1 = self.linear1(x)
    relu = self.relu(linear1)
    linear2 = self.linear2(relu)
    out = self.sigmoid(linear2)
    return out
  
def npnormalize(data):
   normalizedData = (data-np.min(data))/(np.max(data)-np.min(data))
   return normalizedData

# Instantiate a model 
net = SimpleNN(2, 32, 1)
#loss func and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

TrainingClasses = episodesDf["good_or_bad"].to_numpy()

#get input features per training classes
InputData1 = episodesDf["us_viewers_in_millions"].to_numpy()
InputData2 = episodesDf["views"].to_numpy()

InputData1 = npnormalize(InputData1)
InputData2 = npnormalize(InputData2)
inputData = np.column_stack((InputData1, InputData2))

#split training and testing
X_train, X_test, y_train, y_test = train_test_split(inputData, TrainingClasses, test_size=0.2, random_state=42, shuffle=True)

y_test = torch.from_numpy(y_test).to(torch.float32) 
X_test = torch.from_numpy(X_test).to(torch.float32) 

y = torch.from_numpy(y_train).to(torch.float32) 
x = torch.from_numpy(X_train).to(torch.float32) 
y = y.view(-1, 1)
y_test = y_test.view(-1, 1).long()

#print(x)


# The training loop
epochs = 1000

for epoch in range(epochs):
  optimizer.zero_grad()

  # Get model predictions
  y_preds = net(x)

  # Compute the loss by comparing predicted outputs and true labels
  loss = loss_fn(y_preds, y.float())
  #print(f"Epoch {epoch}: training loss: {loss}")

  # Compute the gradients

  loss.backward()

  # Take a step to update (optimize) the parameters
  optimizer.step()


net.eval()

with torch.no_grad():
  outputs = net(X_test)
  predicted_classes = (outputs > 0.5).long()
  # print(predicted_classes)
  # print(y_test)
  zippedout = torch.column_stack((outputs, y_test))
  print(f"zipped out: {zippedout}")
  accuracy = accuracy_score(y_test, predicted_classes)
  precision = precision_score(y_test, predicted_classes)
  recall = recall_score(y_test, predicted_classes)
  f1 = f1_score(y_test, predicted_classes)
  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1: {f1}")