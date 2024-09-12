import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = cosine_similarity(vector1, vector2)
    
    expected_result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    data_points = np.array([[1, 2], [3, 4], [5, 6]])
    query_point = np.array([3.1, 4.1])
    
    result = nearest_neighbor(data_points, query_point)
    
    expected_index = 1  # Since the second point [3, 4] is closest to the query point
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"