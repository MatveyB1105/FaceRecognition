import pickle

file_path = 'embeddings.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

print(data)