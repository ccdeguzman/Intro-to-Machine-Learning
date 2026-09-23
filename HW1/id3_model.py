import pandas as pd
import numpy as np
import sys

# Node for ID3 function
class Node:
    def __init__(self):
        self.feature = None                                 # What feature to split on at this node
        self.label = None                                   # Stores the final answer
        self.children = {}                                  # branches to child nodes

# 3.1 Entropy
def entropy(labels):
    if len(labels) == 0:                                    # return if labels are empty
        return 0
    counts = labels.value_counts()                          # counts occurences
    p = counts / len(labels)                                # gets proportion for each class
    result = -np.sum(p * np.log2(p))
    
    return result

# 3.2 Information Gain
def info_gain(data, feature, target):
    total_ent = entropy(data[target])                       # Getting total entropy of full dataset
    values = data[feature].unique()                         # Getting unique values of the features
    
    # For computing weighted entropy after split
    weighted_ent = 0                  
    for v in values:
        subset = data[data[feature] == v]
        weight = len(subset) / len(data)
        weighted_ent += weight * entropy(subset[target])
        
    return total_ent - weighted_ent

# 4.2 Implementation: Recursive Building
def id3(data, features, target):
    node = Node()
    
    # Base case 1- If everything in the group has the same label return a single node tree with that label.  It is sorted
    if (len(data[target].unique()) == 1):
        node.label = data[target].iloc[0]
        return node
    
    # Base case 2 - no more features to split on
    if len(features) == 0:
        node.label = data[target].mode[0]
        return node
    
    # Picking a feature with the highest information gain
    gains = {}
    
    for i in features:
        gain_val = info_gain(data, i, target)
        gains[i] = gain_val
    
    best_feature = None
    best_gain = -1
    
    for feature_name in gains:
        if gains[feature_name] > best_gain:
            best_gain = gains[feature_name]
            best_feature = feature_name
    
    node.feature = best_feature
    
    # Recursively build a subtree for each value of the best feature
    remaining_features = []
    
    for i in features:
        if i != best_feature:
            remaining_features.append(i)
    
    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        child_subtree = id3(subset, remaining_features, target)
        node.children[val] = child_subtree    
    
    return node

# 4.2 Implementation: Visualization
def print_tree(node, indent = 0):
    prefix = "    " * indent                            # giving 4 spaces for level
    
    # If it's a leaf node. Else print the feature
    if node.label is not None:
        return f"{prefix} Label: {node.label}\n"
    
    result = f"{prefix} Feature: {node.feature}\n"
    
    # Print the children
    for val in node.children:
        result += f"{prefix} Value: {val}\n"
        result += print_tree(node.children[val], indent + 1)
    
    return result

def predict(node, sample):
    if node.label is not None:
        return node.label
    val = sample.get(node.feature)
    if val not in node.children:
        return None
    return predict(node.children[val], sample)


def main():
    df = pd.read_csv("job_offers.csv")
    target = "Accept Offer?"
    features = [c for c in df.columns if c!= target]

    tree = id3(df, features, target)                            # creating tree
    # print("Tree built! Root feature:", tree.feature)

    # for f in features:
        # print(f"Gain({f}): {info_gain(df, f, target):.4f}")

    print_tree(tree)
    
    # Test and write into output.txt
    with open("output.txt", "w") as file:
        file.write("Tree:\n")
        file.write(print_tree(tree))
        file.write("\nPrediction for (Remote: Yes, Salary: No, Tech: Legacy, Benefits: Poor)\n")
        test = {"Remote": "Yes", "Salary High": "No", "Tech Stack": "Legacy", "Benefits": "Poor"}
        file.write(f"Prediction: {predict(tree, test)}\n")
        
    
if __name__ == "__main__":
    main()