import numpy as np

from traffic_data_engine.mine import load_index, rank_by_similarity, save_index


def test_rank_by_similarity_orders_by_cosine():
    index = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [np.sqrt(0.5), np.sqrt(0.5)],
        ]
    )
    query = np.array([1.0, 0.0])
    order, sims = rank_by_similarity(query, index, top_k=3)
    assert order.tolist() == [0, 2, 1]
    assert np.allclose(sims, [1.0, np.sqrt(0.5), 0.0])


def test_rank_by_similarity_clamps_top_k():
    index = np.array([[1.0, 0.0]])
    order, sims = rank_by_similarity(np.array([1.0, 0.0]), index, top_k=10)
    assert order.tolist() == [0]
    assert len(sims) == 1


def test_index_roundtrip(tmp_path):
    embeddings = np.random.default_rng(0).random((3, 4))
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    index_path = tmp_path / "index.npz"
    save_index(index_path, embeddings, paths)
    restored_embeddings, restored_paths = load_index(index_path)
    assert np.allclose(restored_embeddings, embeddings)
    assert restored_paths == paths
