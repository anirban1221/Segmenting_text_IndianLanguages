import torch
import numpy as np
from transformers import AutoTokenizer

class TextSegmentModel:
    def __init__(self, model_path: str, tokenizer_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def get_embeddings(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state.cpu().numpy().squeeze()

    def _dummy_vector(self, text: str) -> torch.Tensor:
        raise NotImplementedError("Use get_embeddings instead.")