import numpy as np

from src.protection import l2_normalize


def evaluate_retrieval(query_embeddings, corpus_embeddings):
    """
    Evaluate retrieval performance under the same-index gold setting.

    Query i's correct document is Corpus i.

    Args:
        query_embeddings: numpy array of shape (N, D)
        corpus_embeddings: numpy array of shape (N, D)

    Returns:
        Dictionary with Hit@1, Hit@5, and MRR.
    """
    query_embeddings = l2_normalize(query_embeddings)
    corpus_embeddings = l2_normalize(corpus_embeddings)

    similarity_matrix = query_embeddings @ corpus_embeddings.T
    num_queries = similarity_matrix.shape[0]

    hit_at_1 = 0
    hit_at_5 = 0
    reciprocal_rank_sum = 0.0

    for i in range(num_queries):
        similarities = similarity_matrix[i]
        gold_score = similarities[i]

        rank = int(np.sum(similarities > gold_score) + 1)

        if rank == 1:
            hit_at_1 += 1

        if rank <= 5:
            hit_at_5 += 1

        reciprocal_rank_sum += 1.0 / rank

    return {
        "Hit@1": hit_at_1 / num_queries,
        "Hit@5": hit_at_5 / num_queries,
        "MRR": reciprocal_rank_sum / num_queries,
    }


def evaluate_retrieval_from_files(query_path, corpus_path):
    """
    Load query and corpus embeddings from .npy files and evaluate retrieval.
    """
    query_embeddings = np.load(query_path).astype(np.float32)
    corpus_embeddings = np.load(corpus_path).astype(np.float32)

    return evaluate_retrieval(
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
    )