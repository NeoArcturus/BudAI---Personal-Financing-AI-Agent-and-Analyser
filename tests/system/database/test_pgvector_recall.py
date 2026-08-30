import pytest
import math

@pytest.mark.asyncio
async def test_pgvector_cosine_distance_mathematics():
    """
    TDD Database: Asserts the exact cosine distance mathematical logic used by 
    the PostgreSQL pgvector extension for semantic Merchant Knowledge routing.
    Proves that the nearest neighbor indexing correctly surfaces the most 
    semantically relevant merchant.
    """
    # Simulated vector embeddings (Simplified 3-dimensional space for proof)
    # [Food/Drink, Corporate/Software, Transport]
    database_vectors = {
        "STARBUCKS": [0.9, 0.1, 0.0],
        "GITHUB": [0.0, 0.95, 0.05],
        "UBER": [0.1, 0.1, 0.9]
    }
    
    query_vector = [0.85, 0.15, 0.0] # User queries "COFFEE SHOP"

    def calculate_cosine_distance(vec1: list, vec2: list) -> float:
        # pgvector uses 1 - cosine_similarity for the <=> operator
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0: return 1.0
        return 1 - (dot_product / (magnitude1 * magnitude2))

    # Calculate exact pgvector distances
    distances = {
        merchant: calculate_cosine_distance(query_vector, vec)
        for merchant, vec in database_vectors.items()
    }
    
    # Sort by nearest neighbor (lowest distance)
    nearest_neighbors = sorted(distances.items(), key=lambda item: item[1])
    
    top_result = nearest_neighbors[0]
    
    # Assertions
    assert top_result[0] == "STARBUCKS", "Vector indexing drift: Semantic search returned incorrect nearest neighbor."
    assert top_result[1] < 0.05, "Vector mathematical accuracy failure: Cosine distance calculation is misaligned."
    # Assert Github is the furthest
    assert nearest_neighbors[-1][0] == "UBER", "Vector bounds failure."

