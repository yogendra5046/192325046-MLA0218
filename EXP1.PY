import numpy as np
import pandas as pd
def find_s_algorithm(training_data):
    """Implements the Find-S algorithm for learning the most specific hypothesis."""
    # Initialize the hypothesis with the first positive example
    hypothesis = None
    
    for example in training_data:
        attributes, label = example[:-1], example[-1]
        
        if label == "Yes":  # Consider only positive examples
            if hypothesis is None:
                hypothesis = list(attributes)  # First positive example initializes the hypothesis
            else:
                # Generalize the hypothesis
                for i in range(len(hypothesis)):
                    if hypothesis[i] != attributes[i]:
                        hypothesis[i] = '?'  # Replace differing attributes with '?'
    
    return hypothesis

# Sample training data (Attributes: Sky, Temp, Humidity, Wind, Water, Forecast, PlayTennis)
training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

# Convert to DataFrame for readability
columns = ["Sky", "Temp", "Humidity", "Wind", "Water", "Forecast", "PlayTennis"]
df = pd.DataFrame(training_data, columns=columns)

# Run the Find-S algorithm
hypothesis = find_s_algorithm(training_data)

print("Final Hypothesis:", hypothesis)