import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Defining the LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden layer dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        # Building LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Linear Output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # One time step
        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Return last linear layer output as prediction
        prediction = self.linear(lstm_out[:, -1, :]) 
        return prediction


if __name__ == "__main__":
	# ------------------------- Data Preprocessing	
	# Read File
	df = pd.read_csv('Dataset.csv',encoding="utf-8")

	# Calculate SMA30
	adjClose = df['Adj Close']
	MA_window = 30
	SMA_30 = adjClose.rolling(window=MA_window).mean()[MA_window-1:]
	dataSize = len(adjClose)

	# Normalize Data (SMA30) & adjClose
	# Here we simply divide the stock price by Maxima, rather than using standarization or Min Max Scaler
	SMA_30_normalized = (SMA_30/max(SMA_30.values)).values
	adjClose_normalized = (adjClose/max(adjClose.values)).values.reshape(-1,1)

	# Create Input/Output Set
	input_window = 10
	input_output_set_size = dataSize-input_window-28
	input_seq = []
	output = []

	for i in range(input_output_set_size):
		input_seq.append(adjClose_normalized[i:input_window+i])
		output.append(SMA_30_normalized[i+input_window-1:i+input_window])

	# Split Data to Training/Validation/Testing Data
	train_data_ratio, val_data_ratio, test_data_ratio = 0.7, 0.15, 0.15
	train_input = torch.FloatTensor(input_seq[:round(input_output_set_size*train_data_ratio)])
	train_labels = torch.FloatTensor(output[:round(input_output_set_size*train_data_ratio)])
	val_input = torch.FloatTensor(input_seq[round(input_output_set_size*train_data_ratio):-round(input_output_set_size*test_data_ratio)])
	val_labels = torch.FloatTensor(output[round(input_output_set_size*train_data_ratio):-round(input_output_set_size*test_data_ratio)])
	test_input = torch.FloatTensor(input_seq[-round(input_output_set_size*test_data_ratio):])
	test_labels = torch.FloatTensor(output[-round(input_output_set_size*test_data_ratio):])
	trainSize, valSize, testSize  = len(train_input), len(val_input), len(test_input)

	# ------------------------- Builiding LSTM Model
	input_dim = 1
	hidden_dim = 32
	num_layers = 2 
	output_dim = 1
	model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

	# ------------------------- Training the Model
	loss_fn = torch.nn.MSELoss(size_average=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)	
	num_epochs = 3000 
	min_val_loss = np.Inf	
	train_loss_overall = []
	val_loss_overall = []
	best_epoch = 0

	for t in range(num_epochs):
		model.train()
		# Forward pass
		y_train_pred = model(train_input)
		train_loss = loss_fn(y_train_pred, train_labels)
		# Zero out gradient, else they will accumulate between epochs
		optimizer.zero_grad()
		# Backward pass
		train_loss.backward()
		# Update parameters
		optimizer.step()

		# Validation
		model.eval()
		y_val_pred = model(val_input)
		val_loss = loss_fn(y_val_pred, val_labels)
		print('-------------------------')
		print('Training Loss {}: {:.8f}'.format(t, train_loss.item()))
		print('Validation Loss {}: {:.5f}'.format(t, val_loss.item()))
		train_loss_overall.append(train_loss.item())
		val_loss_overall.append(val_loss.item())		

		if val_loss.item() < min_val_loss and val_loss.item() < 0.0005:
			min_val_loss = val_loss.item()
			best_epoch = t
			print('------------------------- Small Validation Loss: {:.5f}'.format(min_val_loss))
			torch.save(model.state_dict(), 'result/Model_{:.5f}_{}.pt'.format(min_val_loss,best_epoch))
			

	# ------------------------- Making Predictions
	# Loading best Model
	model.load_state_dict(torch.load('result/Model_{:.5f}_{}.pt'.format(min_val_loss,best_epoch)))
	model.eval()

	y_test_pred = model(test_input)
	test_loss = loss_fn(y_test_pred, test_labels)
	print('Testing Loss: {:.5f}'.format(test_loss.item()))

	actual_prediction = y_test_pred.detach().numpy() * max(SMA_30.values)

	# 在伺服器上繪圖需切換backend
	plt.switch_backend('agg')

	x_all = np.arange(0, dataSize, 1)
	x_all_SMA30 = np.arange(29, dataSize, 1)
	x_predict = np.arange(trainSize+valSize+input_window+28, dataSize, 1)

	# Figure: Full SMA30 Data
	plt.figure(figsize=(50,50))
	plt.title('SMA30 vs Time',fontsize=80)
	plt.ylabel('SMA30',fontsize=60)
	plt.grid(True)
	plt.autoscale(axis='x', tight=True)
	plt.plot(x_all_SMA30, SMA_30,label='SMA_30')
	plt.plot(x_predict,actual_prediction,label='SMA_30_LSTM')	
	plt.legend(loc='upper right', fontsize=42)
	plt.savefig('result/SMA30_final.png'.format(min_val_loss,best_epoch))
	plt.clf()	

	# Figure: Zoom in the Prediction Period
	plt.figure(figsize=(50,50))
	plt.title('SMA30 vs Time',fontsize=80)
	plt.ylabel('SMA30',fontsize=60)
	plt.grid(True)
	plt.autoscale(axis='x', tight=True)
	plt.plot(x_predict, SMA_30[-testSize:],label='SMA_30')
	plt.plot(x_predict,actual_prediction,label='SMA_30_LSTM')	
	plt.legend(loc='upper right', fontsize=42)
	plt.savefig('result/SMA30_final_zoom.png'.format(min_val_loss,best_epoch))
	plt.clf()		

	# Plot Train/Val Loss
	plt.title('Training Loss vs Validation Loss')
	plt.ylabel('Loss')
	plt.grid(True)
	plt.autoscale(axis='x', tight=True)
	plt.plot(train_loss_overall, label='Training Loss')
	plt.plot(val_loss_overall, label='Validation Loss')	
	plt.legend(loc='upper right', fontsize=36)
	plt.savefig('result/Loss_final.png'.format(min_val_loss,best_epoch))
	plt.clf()

	




