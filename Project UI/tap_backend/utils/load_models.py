import pickle


def load_model(path):
    # Open the pickle file in read-binary mode ('rb')
    with open(path, 'rb') as file:
        data = pickle.load(file)  # Load the pickled data

    # Display the data
    print(data)
    return data


