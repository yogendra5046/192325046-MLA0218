import pandas as pd
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename)
    return df.iloc[:, :-1].values, df.iloc[:, -1].values  # Features and labels

def initialize_hypotheses(X, y):
    num_attributes = X.shape[1]
    S = None
    for instance, label in zip(X, y):
        if label == 'Yes':
            S = instance.copy()
            break
    if S is None:
        S = ['?'] * num_attributes  # Default if no positive example exists
    G = [['?'] * num_attributes]  # Most general hypothesis
    return S, G

def update_hypotheses(S, G, instance, label):
    num_attributes = len(S)
    
    if label == 'Yes':  # Positive example
        for i in range(num_attributes):
            if S[i] != instance[i]:
                S[i] = '?'
        
        G = [g for g in G if all(g[i] == '?' or g[i] == S[i] for i in range(num_attributes))]
    else:  # Negative example
        new_G = []
        for g in G:
            for i in range(num_attributes):
                if g[i] == '?':
                    new_hypothesis = g.copy()
                    new_hypothesis[i] = S[i]
                    new_G.append(new_hypothesis)
        G = [g for g in new_G if any(g[i] != '?' for i in range(num_attributes))]
    
    return S, G

def candidate_elimination(filename):
    X, y = load_data(filename)
    S, G = initialize_hypotheses(X, y)
    
    for instance, label in zip(X, y):
        S, G = update_hypotheses(S, G, instance, label)
        print(f"After instance {instance} ({label}):")
        print(f"Specific Hypothesis: {S}")
        print(f"General Hypotheses: {G}\n")
        
    return S, G

if __name__ == "__main__":
    filename = "training_data.csv"  # Ensure this CSV file exists with proper format
    S_final, G_final = candidate_elimination(filename)
    print("Final Specific Hypothesis:", S_final)
    print("Final General Hypotheses:", G_final)
