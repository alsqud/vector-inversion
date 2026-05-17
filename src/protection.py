import numpy as np
from sklearn.decomposition import PCA


def l2_normalize(embeddings, eps=1e-12):
    """
    L2-normalize embedding vectors.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings / np.maximum(norm, eps)


def add_gaussian_noise(embeddings, sigma, seed):
    """
    Add Gaussian noise to embeddings and normalize them.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if float(sigma) == 0:
        return embeddings.copy()

    rng = np.random.RandomState(int(seed))
    noise = rng.randn(*embeddings.shape).astype(np.float32) * float(sigma)

    noisy_embeddings = embeddings + noise
    noisy_embeddings = l2_normalize(noisy_embeddings)

    return noisy_embeddings


def fit_pca(train_embeddings, pca_dim, seed=42):
    """
    Fit PCA using training embeddings.
    """
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)

    pca = PCA(
        n_components=int(pca_dim),
        random_state=int(seed),
    )

    pca.fit(train_embeddings)

    explained_variance = float(np.sum(pca.explained_variance_ratio_))

    return pca, explained_variance


def apply_pca(query_embeddings, corpus_embeddings, pca):
    """
    Apply PCA to query and corpus embeddings for retrieval evaluation.
    """
    query_pca = pca.transform(query_embeddings).astype(np.float32)
    corpus_pca = pca.transform(corpus_embeddings).astype(np.float32)

    query_pca = l2_normalize(query_pca)
    corpus_pca = l2_normalize(corpus_pca)

    return query_pca, corpus_pca


def apply_pca_for_inversion(text_embeddings, pca):
    """
    Apply PCA and inverse transform back to the original embedding dimension.

    This is used when the inversion attacker expects the original embedding dimension.
    """
    text_pca = pca.transform(text_embeddings).astype(np.float32)
    text_pca = l2_normalize(text_pca)

    reconstructed_embeddings = pca.inverse_transform(text_pca).astype(np.float32)
    reconstructed_embeddings = l2_normalize(reconstructed_embeddings)

    return reconstructed_embeddings


def apply_pca_noise_for_retrieval(
    query_embeddings,
    corpus_embeddings,
    pca,
    sigma,
    seed,
):
    """
    Apply PCA and Gaussian noise for retrieval evaluation.
    """
    query_pca, corpus_pca = apply_pca(
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        pca=pca,
    )

    if float(sigma) > 0:
        corpus_pca = add_gaussian_noise(
            embeddings=corpus_pca,
            sigma=sigma,
            seed=seed,
        )

    return query_pca, corpus_pca


def apply_pca_noise_for_inversion(
    text_embeddings,
    pca,
    sigma,
    seed,
):
    """
    Apply PCA, Gaussian noise, and inverse PCA transform for inversion evaluation.
    """
    text_pca = pca.transform(text_embeddings).astype(np.float32)
    text_pca = l2_normalize(text_pca)

    if float(sigma) > 0:
        text_pca = add_gaussian_noise(
            embeddings=text_pca,
            sigma=sigma,
            seed=seed,
        )

    reconstructed_embeddings = pca.inverse_transform(text_pca).astype(np.float32)
    reconstructed_embeddings = l2_normalize(reconstructed_embeddings)

    return reconstructed_embeddings