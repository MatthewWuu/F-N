# -*- coding: utf-8 -*-

import os
import csv
import hypernetx as hnx
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

#############################################
# Part 1: Build Hypergraph from DietaryRecall CSV Files
#############################################
def build_meal_hypergraph_with_keys(folder_path):
    """
    Searches for all CSV files in the specified folder (filenames starting with 'DR' or matching extra files)
    and groups rows by (SEQN, DRxMC, DRx_040Z). For each group, it collects the corresponding food codes
    (DR1IFDCD/DR2IFDCD). Returns a dictionary: key -> (SEQN, DRxMC, DRx_040Z), value -> set of food codes.
    """
    extra_files = ["P_DR1IFF.csv", "P_DR1IFF2.csv"]
    csv_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            if fname.startswith("DR") or fname in extra_files:
                csv_files.append(os.path.join(folder_path, fname))
    
    print("Will process the following CSV files:")
    for f in csv_files:
        print("  ", f)
    
    meal_dict = defaultdict(set)
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"Processing {os.path.basename(csv_file)}"):
                seqn = row.get("SEQN", "").strip()
                # Try DR1MC first, then DR2MC
                meal_code = row.get("DR1MC", "").strip() or row.get("DR2MC", "").strip()
                # Try DR1_040Z first, then DR2_040Z
                loc_code = row.get("DR1_040Z", "").strip() or row.get("DR2_040Z", "").strip()
                # Try DR1IFDCD first, then DR2IFDCD
                food_code = row.get("DR1IFDCD", "").strip() or row.get("DR2IFDCD", "").strip()
                if not seqn or not meal_code or not loc_code or not food_code:
                    continue
                key = (seqn, meal_code, loc_code)
                meal_dict[key].add(food_code)
    return meal_dict

def build_hypergraph_and_edge_features(meal_dict):
    """
    Constructs a hypergraph using hypernetx, where each hyperedge is formed by the union of the grouping key
    (SEQN, DRxMC, DRx_040Z) and the corresponding food codes.
    Also returns a dictionary (edge_keys) mapping hyperedge IDs to their grouping keys.
    """
    edges_dict = {}
    edge_keys = {}
    idx = 0
    for key, foods in meal_dict.items():
        seqn, meal_code, loc_code = key
        # Hyperedge nodes: union of the three elements of the grouping key and the food codes
        edge_nodes = set([seqn, meal_code, loc_code]) | foods
        edge_id = f"meal_{idx}"
        edges_dict[edge_id] = edge_nodes
        edge_keys[edge_id] = key
        idx += 1
    H = hnx.Hypergraph(edges_dict)
    return H, edge_keys

#############################################
# Part 2: Load Demographic Data from Excel (.xlsx)
#############################################
def load_demographic_data(demo_folder, suffix):
    """
    Loads demographic data from an Excel file based on the naming rule (e.g., DEMO_A.xlsx)
    and returns a dictionary mapping SEQN to the corresponding demographic information.
    """
    demo_filename = f"DEMO{suffix}.xlsx"
    demo_path = os.path.join(demo_folder, demo_filename)
    print("Loading demographic data from:", demo_path)
    df = pd.read_excel(demo_path)
    demo_dict = df.set_index("SEQN").to_dict(orient="index")
    return demo_dict

#############################################
# Part 3: Build Mixed Graph using PyG
#############################################
def build_pyg_data_from_hypergraph(hypergraph, demo_dict):
    """
    Constructs a PyTorch Geometric Data object from the hypergraph and attaches demographic information to SEQN nodes.
    Steps:
      1. Create a mapping from all hypergraph nodes to unique indices.
      2. Convert hyperedges to graph edges using clique expansion (to form an undirected graph).
      3. Attach demographic information to nodes that are considered SEQN (assumed numeric strings of length <= 8).
    Returns:
      data: A PyG Data object with attributes:
            - edge_index: Tensor of shape [2, num_edges]
            - node_mapping: Dictionary mapping node names to indices
            - demo_attr: Dictionary mapping node index to demographic information
    """
    # Build node mapping
    node_list = list(hypergraph.nodes)
    node2idx = {node: i for i, node in enumerate(node_list)}
    
    # Build edge list using clique expansion for hyperedges
    edge_list = []
    for edge, node_set in hypergraph.incidence_dict.items():
        nodes_in_edge = list(node_set)
        for i in range(len(nodes_in_edge)):
            for j in range(i+1, len(nodes_in_edge)):
                u = node2idx[nodes_in_edge[i]]
                v = node2idx[nodes_in_edge[j]]
                edge_list.append((u, v))
                edge_list.append((v, u))
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Attach demographic data to SEQN nodes (assuming SEQN is a numeric string with length <= 8)
    demo_attr = {}
    for node, idx in node2idx.items():
        if node.replace('.', '', 1).isdigit() and len(node) <= 8:
            if node in demo_dict:
                demo_attr[idx] = demo_dict[node]
    
    data = Data(edge_index=edge_index)
    data.node_mapping = node2idx
    data.demo_attr = demo_attr
    return data

#############################################
# Part 4: Hyperedge Embedding (Inner-embedded Feature Design)
#############################################
def build_mapping(values):
    """Build a mapping from unique discrete values to indices."""
    unique_vals = sorted(list(set(values)))
    mapping = {val: i for i, val in enumerate(unique_vals)}
    return mapping

class HyperedgeEmbedding(nn.Module):
    def __init__(self, num_seqn, num_meal, num_loc, d_seqn, d_meal, d_loc, fusion_out_dim):
        """
        Initializes embedding layers for SEQN, DRxMC, and DRx_040Z,
        and defines a fusion layer to combine the concatenated vectors into the final hyperedge embedding.
        """
        super(HyperedgeEmbedding, self).__init__()
        self.embed_seqn = nn.Embedding(num_seqn, d_seqn)
        self.embed_meal = nn.Embedding(num_meal, d_meal)
        self.embed_loc  = nn.Embedding(num_loc, d_loc)
        self.fusion_layer = nn.Linear(d_seqn + d_meal + d_loc, fusion_out_dim)
    
    def forward(self, seqn_idx, meal_idx, loc_idx):
        seqn_emb = self.embed_seqn(seqn_idx)
        meal_emb = self.embed_meal(meal_idx)
        loc_emb  = self.embed_loc(loc_idx)
        combined = torch.cat([seqn_emb, meal_emb, loc_emb], dim=-1)
        out = self.fusion_layer(combined)
        return out

#############################################
# Part 5: Add Nutrition Nodes to PyG Graph
#############################################
def add_nutrition_nodes(pyg_data, folder_path):
    """
    Reads nutritional information from DietaryRecall CSV files,
    creates a new nutrition node for each valid nutritional record, and
    connects the nutrition node with the corresponding food node via bidirectional edges.
    
    Parameters:
      pyg_data: The existing PyG Data object whose node_mapping includes nodes from the hypergraph (including food nodes).
      folder_path: The folder path containing the DietaryRecall CSV files.
    
    Returns:
      The updated pyg_data object with added nutrition nodes, a nutrition_attr attribute, and the corresponding edges.
    """
    extra_files = ["P_DR1IFF.csv", "P_DR1IFF2.csv"]
    csv_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            if fname.startswith("DR") or fname in extra_files:
                csv_files.append(os.path.join(folder_path, fname))
    
    current_num_nodes = pyg_data.num_nodes
    node_mapping = pyg_data.node_mapping.copy()
    
    new_edges = []         # To store edges between nutrition nodes and food nodes
    nutrition_attr = {}    # To store nutritional information for each nutrition node
    nutrition_counter = 0  # Counter to generate nutrition node names
    
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"Processing Nutrition in {os.path.basename(csv_file)}"):
                # Get food code: try DR1IFDCD first, then DR2IFDCD
                food_code = row.get("DR1IFDCD", "").strip() or row.get("DR2IFDCD", "").strip()
                if not food_code:
                    continue
                
                # Assume nutritional fields exist: Energy, Protein, Fat, Carbohydrate
                nutrition_info = {}
                for col in ["DR1IGRMS", "DR1IKCAL", "DR1IPROT", "DR1ICARB", "DR1ISUGR", "DR1IFIBE",
    "DR1ITFAT", "DR1ISFAT", "DR1IMFAT", "DR1IPFAT", "DR1ICHOL", "DR1IATOC",
    "DR1IATOA", "DR1IRET", "DR1IVARA", "DR1IACAR", "DR1IBCAR", "DR1ICRYP",
    "DR1ILYCO", "DR1ILZ", "DR1IVB1", "DR1IVB2", "DR1INIAC", "DR1IVB6",
    "DR1IFOLA", "DR1IFA", "DR1IFF", "DR1IFDFE", "DR1IVB12", "DR1IB12A",
    "DR1IVC", "DR1IVK", "DR1ICALC", "DR1IPHOS", "DR1IMAGN", "DR1IIRON",
    "DR1IZINC", "DR1ICOPP", "DR1ISODI", "DR1IPOTA", "DR1ISELE", "DR1ICAFF",
    "DR1ITHEO", "DR1IALCO", "DR1IMOIS", "DR1IS040", "DR1IS060", "DR1IS080",
    "DR1IS100", "DR1IS120", "DR1IS140", "DR1IS160", "DR1IS180", "DR1IM161",
    "DR1IM181", "DR1IM201", "DR1IM221", "DR1IP182", "DR1IP183", "DR1IP184",
    "DR1IP204", "DR1IP205", "DR1IP225", "DR1IP226"]:
                    val = row.get(col, "").strip()
                    if val:
                        nutrition_info[col] = val
                if not nutrition_info:
                    continue
                
                # Check if the corresponding food node exists in the graph
                if food_code not in node_mapping:
                    # If not found, skip (or optionally add the food node); here we choose to skip.
                    continue
                
                food_node_idx = node_mapping[food_code]
                
                # Create a new nutrition node
                nutrition_node_name = f"nutrition_{nutrition_counter}"
                nutrition_node_idx = current_num_nodes
                current_num_nodes += 1
                nutrition_counter += 1
                
                node_mapping[nutrition_node_name] = nutrition_node_idx
                
                # Add bidirectional edge between the nutrition node and the food node
                new_edges.append((nutrition_node_idx, food_node_idx))
                new_edges.append((food_node_idx, nutrition_node_idx))
                
                # Save the nutritional information for this nutrition node
                nutrition_attr[nutrition_node_idx] = nutrition_info
    
    # Update the node mapping and total node count in pyg_data
    pyg_data.node_mapping = node_mapping
    pyg_data.num_nodes = current_num_nodes
    
    # Merge the new edges into the existing edge_index
    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        pyg_data.edge_index = torch.cat([pyg_data.edge_index, new_edge_tensor], dim=1)
    
    pyg_data.nutrition_attr = nutrition_attr
    print(f"Added {nutrition_counter} nutrition nodes, creating {len(new_edges)//2} nutrition-food edges")
    return pyg_data

#############################################
# Main Execution
#############################################
if __name__ == "__main__":
    # Part 1: Build the hypergraph and grouping keys
    folder_path = "/Users/wujunwei/MMKG/F+N/NHANES data/NHANES/DietaryRecall"  # Change to the actual data path
    meal_dict = build_meal_hypergraph_with_keys(folder_path)
    hypergraph, edge_keys = build_hypergraph_and_edge_features(meal_dict)
    
    print("\n=== Hypergraph Info ===")
    print("Number of edges:", len(hypergraph.edges))
    print("Number of nodes:", len(hypergraph.nodes))
    sorted_edges = sorted(hypergraph.incidence_dict.items(), key=lambda x: int(x[0].split('_')[1]))
    for e_id, node_set in sorted_edges[:100]:
        key = edge_keys.get(e_id, ("N/A", "N/A", "N/A"))
        print(f"{e_id} -> Nodes: {node_set}")
        print(f"    Grouping key (SEQN, DRxMC, DRx_040Z): {key}")
    
    # Part 2: Load demographic data
    demo_suffix = "_A"  # Adjust according to the actual file naming
    demo_folder = "/Users/wujunwei/MMKG/F+N/NHANES data/NHANES/demographics"  # Change to the actual demographic data path
    demo_data = load_demographic_data(demo_folder, demo_suffix)
    
    # Part 3: Build the PyG mixed graph (with demographic information attached)
    pyg_data = build_pyg_data_from_hypergraph(hypergraph, demo_data)
    print("\n=== PyG Mixed Graph Info ===")
    print("Number of nodes in PyG Data:", pyg_data.num_nodes)
    print("Number of edges in PyG Data:", pyg_data.edge_index.size(1))
    print("\nSample node demographic attributes (up to 100):")
    count = 0
    for idx in list(pyg_data.node_mapping.values()):
        if idx in pyg_data.demo_attr:
            print(f"Node index: {idx}, Demographic attributes: {pyg_data.demo_attr[idx]}")
            count += 1
            if count >= 100:
                break
    
    # Part 4: Hyperedge embedding (optional)
    seqn_vals = [key[0] for key in edge_keys.values()]
    meal_vals = [key[1] for key in edge_keys.values()]
    loc_vals  = [key[2] for key in edge_keys.values()]
    seqn2idx = build_mapping(seqn_vals)
    meal2idx = build_mapping(meal_vals)
    loc2idx  = build_mapping(loc_vals)
    
    print("\nMapping dictionaries:")
    print("SEQN mapping:", seqn2idx)
    print("Meal mapping:", meal2idx)
    print("Location mapping:", loc2idx)
    
    d_seqn = 8
    d_meal = 4
    d_loc  = 8
    fusion_out_dim = 16
    embed_model = HyperedgeEmbedding(num_seqn=len(seqn2idx),
                                     num_meal=len(meal2idx),
                                     num_loc=len(loc2idx),
                                     d_seqn=d_seqn,
                                     d_meal=d_meal,
                                     d_loc=d_loc,
                                     fusion_out_dim=fusion_out_dim)
    
    hyperedge_embeddings = {}
    count = 0
    for e_id, key in edge_keys.items():
        seqn, meal, loc = key
        seqn_idx = torch.tensor([seqn2idx[seqn]], dtype=torch.long)
        meal_idx = torch.tensor([meal2idx[meal]], dtype=torch.long)
        loc_idx  = torch.tensor([loc2idx[loc]], dtype=torch.long)
        emb = embed_model(seqn_idx, meal_idx, loc_idx)
        hyperedge_embeddings[e_id] = emb
        print(f"{e_id} Grouping key: {key}, Embedding: {emb.detach().numpy()}")
        count += 1
        if count >= 100:
            break 
    
    # Part 5: Add nutritional information to the graph (connecting nutrition nodes with food nodes)
    pyg_data = add_nutrition_nodes(pyg_data, folder_path)
    
    # Part 6: Compute graph density and sparsity
    V = pyg_data.num_nodes
    edge_set = set()
    num_edges_total = pyg_data.edge_index.size(1)
    for i in range(num_edges_total):
        u = pyg_data.edge_index[0, i].item()
        v = pyg_data.edge_index[1, i].item()
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    E = len(edge_set)
    max_edges = V * (V - 1) / 2
    density = E / max_edges if max_edges > 0 else 0
    sparsity = 1 - density
    print("\n=== Graph Density and Sparsity ===")
    print(f"Graph density: {density:.6f}")
    print(f"Graph sparsity: {sparsity:.6f}")
