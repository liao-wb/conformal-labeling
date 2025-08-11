import pickle

# Load the pickle file
with open('/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/llm/result/prediction_results.pkl', 'rb') as f:
    data = pickle.load(f)

# Print all keys
print("Keys in the pickle file:", data.keys())

# Print the full content
print("\nFull content:")
for key, value in data.items():
    print(f"\n{key}:")
    print(value)