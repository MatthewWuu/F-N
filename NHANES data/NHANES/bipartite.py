import pandas as pd
import torch
from torch_geometric.data import Data


file_path = "/Users/wujunwei/MMKG/DATA/NHANES/DR1IFF.xlsx"
df = pd.read_excel(file_path)

data = df[["SEQN", "DR1IFDCD", "DR1IGRMS"]].dropna()

user_ids = data["SEQN"].unique()  # user
food_ids = data["DR1IFDCD"].unique()  # food ID

user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
food_to_index = {food_id: i + len(user_ids) for i, food_id in enumerate(food_ids)}

edges = []
weights = []
for _, row in data.iterrows():
    user_idx = user_to_index[row["SEQN"]]
    food_idx = food_to_index[row["DR1IFDCD"]]
    edges.append([user_idx, food_idx])
    weights.append(row["DR1IGRMS"])

edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, num_edges]
edge_weight = torch.tensor(weights, dtype=torch.float)  # edge weight

num_users = len(user_ids)
num_foods = len(food_ids)
num_nodes = num_users + num_foods


node_features = torch.ones((num_nodes, 1), dtype=torch.float)

graph_data = Data(edge_index=edge_index, edge_attr=edge_weight, x=node_features)

print(graph_data)

# Calculate user and food node counts
num_users = len(user_ids)
num_foods = len(food_ids)

# Print the results
print(f"Number of user nodes: {num_users}")
print(f"Number of food nodes: {num_foods}")
print(f"Total number of nodes: {graph_data.num_nodes}")
print(f"Total number of edges: {graph_data.num_edges}")
print(f"Edge attributes (weights) sample: {graph_data.edge_attr[:5]}")


#calculate the density
density = graph_data.num_edges / (graph_data.num_nodes * graph_data.num_nodes)
print(f"Graph Density: {density:.6f}")


