import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert data to a list of lists (instead of NumPy array)
data = [np.array(seq) for seq in data_dict['data']]
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Convert x_train and x_test to lists of lists (required for RandomForestClassifier)
x_train_list = [seq.tolist() for seq in x_train]
x_test_list = [seq.tolist() for seq in x_test]

# Find the maximum length of the sequences
max_length = max(len(seq) for seq in x_train_list)

# Pad the sequences to the same length
padded_x_train = []
for seq in x_train_list:
    padded_seq = np.pad(seq, (0, max_length - len(seq)), mode='constant')
    padded_x_train.append(padded_seq)

padded_x_test = []
for seq in x_test_list:
    padded_seq = np.pad(seq, (0, max_length - len(seq)), mode='constant')
    padded_x_test.append(padded_seq)

# Convert the padded lists to NumPy arrays
padded_x_train = np.array(padded_x_train)
padded_x_test = np.array(padded_x_test)

# Create and train the model
model = RandomForestClassifier()
model.fit(padded_x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(padded_x_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(accuracy * 100))

# Calculate the precision score
precision = precision_score(y_test, y_predict, average='weighted')
print('Precision: {:.3f}'.format(precision))

# Calculate the recall score
recall = recall_score(y_test, y_predict, average='weighted')
print('Recall: {:.3f}'.format(recall))

# Print the classification report
print('Classification Report:')
print(classification_report(y_test, y_predict))

# Save the model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)