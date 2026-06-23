import numpy as np

def cosine_distance(a, b):
    """Calculate the cosine distance between 1D arrays a and b."""
    a = np.asarray(a)
    b = np.asarray(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (np.dot(a, b) / (norm_a * norm_b))

class PrototypeDetector:
    """Zero-shot detection module that classifies unlabeled wallets by measuring
    cosine distance to prototype embeddings derived from a small set of confirmed
    wash-trade examples.
    """
    def __init__(self):
        self.wash_prototype = None
        self.legit_prototype = None

    def fit(self, labeled_embeddings, labels):
        """Fit prototypes using labeled embeddings.
        labels == 1 indicates wash trade, labels == 0 indicates legit trade.
        """
        labeled_embeddings = np.asarray(labeled_embeddings)
        labels = np.asarray(labels)
        
        wash_mask = labels == 1
        legit_mask = labels == 0
        
        if np.any(wash_mask):
            self.wash_prototype = labeled_embeddings[wash_mask].mean(axis=0)
        else:
            self.wash_prototype = np.zeros(labeled_embeddings.shape[1])
            
        if np.any(legit_mask):
            self.legit_prototype = labeled_embeddings[legit_mask].mean(axis=0)
        else:
            self.legit_prototype = np.zeros(labeled_embeddings.shape[1])

    def score(self, embedding) -> float:
        """Score an embedding based on its distance to the prototypes.
        Returns a score between 0.0 and 1.0. Higher means more likely wash trade.
        """
        embedding = np.asarray(embedding)
        
        # In case prototypes are not fitted properly
        if self.wash_prototype is None or self.legit_prototype is None:
            return 0.5
            
        d_wash = cosine_distance(embedding, self.wash_prototype)
        d_legit = cosine_distance(embedding, self.legit_prototype)
        
        # Handle case where both distances are 0
        if d_wash + d_legit == 0:
            return 0.5
            
        return d_legit / (d_wash + d_legit)
