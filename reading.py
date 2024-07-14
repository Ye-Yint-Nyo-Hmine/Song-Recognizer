import pickle
import pprint


def pickled_reading(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()

    with open("fingerprints_database.txt", "a") as f:
        pprint.pprint(data, stream = f)

pickled_reading("fingerprint_database.pkl")