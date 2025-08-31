import pickle
from adalflow.core.db import LocalDB
import os

path = "/home/lyr/.adalflow/databases/#home#lyr#solidity-dual-vector-query-driven-dashscope-text-embedding-v4.pkl"

path = "/home/lyr/.adalflow/databases/#home#lyr#solidity-dual-vector-bm25-dashscope-text-embedding-v4.pkl"

db = LocalDB.load_state(path)


for item in db.items:
    print(id(item), item.meta_data["file_path"])

print(len(db.items))
# print(db.transformed_items.keys())
print(len(db.transformed_items[os.path.basename(path)]))

# class A:
#     def __init__(self, x: int):
#         self.x = x

#     def call(self):
#         print(self.x)
#         self.x += 1
#         return self.x


# a = A(1)
# print(id(a))
# import pickle

# pickle.dump(a, open("a.pkl", "wb"))

# with open("a.pkl", "rb") as f:
#     b = pickle.load(f)

# print(id(b))
