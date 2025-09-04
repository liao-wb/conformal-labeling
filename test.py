import json

import numpy as np
import pandas as pd

# Define the path to your .jsonl file
file_path = 'preference_data/gpt-3.5-turbo.calibration.jsonl'

# Create an empty list to store the parsed JSON objects
data = []

# Open the file and process it line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse the JSON from the current line and append to the list
        json_object = json.loads(line)
        data.append(json_object)

#print(data[0]) # Print the first object
data = pd.DataFrame(data)

Y = data["preferences"].to_numpy()
Y = np.array([y['human'] for y in Y]) - 1
probs = data["probs"]
#Y_hat = np.argmax(probs, axis=-1)
Y_hat = np.array([np.argmax(prob, axis=-1) for prob in probs])
confidence = np.array([np.max(prob, axis=-1) for prob in probs])