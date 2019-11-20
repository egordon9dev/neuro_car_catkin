import pickle

with open("td", "rb") as f:
    data = pickle.load(f)
    print(data[0])
    for i in data[0][0][0]:
        print(i, end=" ")
    print("")
