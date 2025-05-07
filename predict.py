import numpy as np
import torch
from model import Classifier

# from deepface import DeepFace
def ocean_predict(dataset):
    model = Classifier()
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))  # Use 'cuda' if GPU
    model.eval()
    dataset = torch.tensor(dataset, dtype=torch.float32)
    result = []
    with torch.no_grad():
        for data in dataset:
            data = data.unsqueeze(0)
            output = model(data)
            result.append(output.tolist())
    return np.mean(result[0],axis=0)
