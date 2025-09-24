import pandas as pd
import numpy as np
import json


def get_data(dataset):
    if dataset == "gpt-3.5-turbo" or dataset == "gpt-4-turbo" or dataset == "mistral-7b-instruct" or dataset == "Qwen3-8B":
        Y, Yhat, confidence, _ = get_preference_data(dataset)
    else:
        data = pd.read_csv("./datasets/" + dataset + '.csv')
        Y = data["Y"].to_numpy()
        confidence = data["confidence"].to_numpy()
        Yhat = None
        if dataset in ['stance', 'misinfo', 'bias', 'sentiment']:
            Yhat = data["Yhat (GPT4o)"].to_numpy()
        else:
            Yhat = data["Yhat"].to_numpy()

    return Y, Yhat, confidence

def get_ood_data(dataset):

        data = pd.read_csv("./datasets/" + dataset + '.csv')
        Y = data["Y"].to_numpy()
        msp_confidence = data["msp_confidence"].to_numpy()
        odin_confidence = data["odin_confidence"].to_numpy()
        energy_confidence = data["energy_confidence"].to_numpy()
        Yhat = None
        if dataset in ['stance', 'misinfo', 'bias', 'sentiment']:
            Yhat = data["Yhat (GPT4o)"].to_numpy()
        else:
            Yhat = data["Yhat"].to_numpy()

        return Y, Yhat, msp_confidence, odin_confidence, energy_confidence

def get_preference_data(dataset):
    calib_file_path = "preference_data/" + dataset + ".calibration.jsonl"
    test_file_path = "preference_data/" + dataset + ".1"


    data = []
    # Open the file and process it line by line
    with open(calib_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON from the current line and append to the list
            json_object = json.loads(line)
            data.append(json_object)

    with open(test_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON from the current line and append to the list
            json_object = json.loads(line)
            data.append(json_object)

    # print(data[0]) # Print the first object
    data = pd.DataFrame(data)
    print(data)
    if dataset == "Qwen3-8B":
        Y = data["choice"]
    else:
        Y = data["preferences"].to_numpy()
        Y = np.array([y['human'] for y in Y]) - 1
    probs = data["probs"]
    # Y_hat = np.argmax(probs, axis=-1)
    Y_hat = np.array([np.argmax(prob, axis=-1) for prob in probs])
    confidence = np.array([np.max(prob, axis=-1) for prob in probs])

    data["confidence"] = confidence
    data["Yhat"] = Y_hat
    data["Y"] = Y
    return Y, Y_hat, confidence, data