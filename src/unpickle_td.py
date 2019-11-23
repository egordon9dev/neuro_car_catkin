import pickle
import msgpack

with open("training_data", "rb") as f:
    data = msgpack.unpack(f)
    print("Found " + str(len(data)) + " data frames")
