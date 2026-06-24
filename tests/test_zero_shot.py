import numpy as np
from detection.zero_shot import cosine_distance, PrototypeDetector

def test_cosine_distance():
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    # Orthogonal vectors, cosine distance should be 1.0
    assert np.isclose(cosine_distance(a, b), 1.0)
    
    a = np.array([1, 1, 0])
    b = np.array([2, 2, 0])
    # Parallel vectors, cosine distance should be 0.0
    assert np.isclose(cosine_distance(a, b), 0.0)
    
    a = np.array([1, 0, 0])
    b = np.array([-1, 0, 0])
    # Opposite vectors, cosine distance should be 2.0
    assert np.isclose(cosine_distance(a, b), 2.0)

def test_prototype_detector_toy_problem():
    # Labeled embeddings
    # Wash trades: vectors pointing mostly in x direction
    wash = np.array([[1.0, 0.1], [0.9, -0.1], [1.1, 0.0]])
    # Legit trades: vectors pointing mostly in y direction
    legit = np.array([[0.1, 1.0], [-0.1, 0.9], [0.0, 1.1]])
    
    embeddings = np.vstack([wash, legit])
    labels = np.array([1, 1, 1, 0, 0, 0])
    
    detector = PrototypeDetector()
    detector.fit(embeddings, labels)
    
    # Expected prototypes
    # Wash prototype ~ [1.0, 0.0]
    # Legit prototype ~ [0.0, 1.0]
    
    # Score a new wash-like vector
    new_wash = np.array([1.0, 0.05])
    wash_score = detector.score(new_wash)
    
    # Score a new legit-like vector
    new_legit = np.array([0.05, 1.0])
    legit_score = detector.score(new_legit)
    
    # Wash score should be higher than legit score
    assert wash_score > legit_score
    
    # Wash score should be close to 1.0, legit score close to 0.0
    assert wash_score > 0.8
    assert legit_score < 0.2
