import unittest
import torch
from torch.nn import functional as F

from rad_retrieval import ChunkedRetriever, RetrievalConfig


class ArrayBank:
    def __init__(self, cls, patches):
        self.cls = cls
        self.patches = patches
        self.layers = list(cls)
        self.n_samples = len(cls[self.layers[0]])
        self.patch_shapes = {k: tuple(v.shape) for k, v in patches.items()}
        self.read_sizes = []

    def read_cls(self, layer, start, stop):
        return self.cls[layer][start:stop].clone()

    def read_patches(self, layer, indices):
        self.read_sizes.append(len(indices))
        return self.patches[layer][indices].clone()


def dense_patch(query, neighbors, positional, radius):
    query = F.normalize(query, dim=-1)
    neighbors = F.normalize(neighbors, dim=-1)
    n, length, channels = neighbors.shape
    if not positional:
        return 1 - (query @ neighbors.reshape(-1, channels).t()).max(dim=-1).values
    grid = int(length ** .5)
    scores = []
    for j in range(length):
        row, col = divmod(j, grid)
        indices = [r * grid + c
                   for r in range(max(0, row-radius), min(grid, row+radius+1))
                   for c in range(max(0, col-radius), min(grid, col+radius+1))]
        local = neighbors[:, indices].reshape(-1, channels)
        scores.append(1 - (query[j:j+1] @ local.t()).max())
    return torch.stack(scores)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)
        self.bank = ArrayBank({i: torch.randn(13, 12) for i in (3,6)},
                              {i: torch.randn(13, 16, 12) for i in (3,6)})

    def test_all_patch_modes_match_dense_for_multiple_chunks(self):
        query = torch.randn(16, 12)
        ids = torch.tensor([0, 2, 4, 8, 10, 12])
        for positional, radius in [(False, 0), (True, 0), (True, 1), (True, 2), (True, 50)]:
            expected = dense_patch(query, self.bank.patches[3][ids], positional, radius)
            for reference_chunk, query_chunk in [(1, 1), (2, 3), (5, 9)]:
                with self.subTest(positional=positional, radius=radius, chunk=reference_chunk):
                    retriever = ChunkedRetriever(self.bank, 'cpu', RetrievalConfig(reference_chunk, query_chunk, 4, 8, 8))
                    actual = retriever.patch_scores(3, query, ids, positional, radius)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)
                    self.assertLessEqual(retriever.peak_reference_rows, reference_chunk)

    def test_image_neighbors_match_full_topk_including_ties(self):
        # Exact ties must go through the full one-query torch.topk, not a new
        # chunk-merge tie breaker that can select different reference images.
        self.bank.cls[6][4:] = self.bank.cls[6][0]
        query = self.bank.cls[6][0:1].clone()
        full = F.normalize(query, dim=-1) @ F.normalize(self.bank.cls[6], dim=-1).t()
        for k in (1, 5, self.bank.n_samples):
            expected = torch.topk(full[0], k).indices
            retriever = ChunkedRetriever(self.bank, 'cpu', RetrievalConfig(cls_chunk_size=4))
            actual = retriever.image_neighbors(query, k)[0]
            self.assertEqual(set(actual.tolist()), set(expected.tolist()))

    def test_layer_fusion_and_query_batch(self):
        patches = [torch.randn(3, 16, 12) for _ in (3,6)]
        cls = [torch.randn(3, 12) for _ in (3,6)]
        retriever = ChunkedRetriever(self.bank, 'cpu', RetrievalConfig(2, 3, 4, 8, 8))
        actual, ids = retriever.score_batch(patches, cls, 7, True, 1, [1.,3.])
        expected = []
        for b in range(3):
            expected.append(sum(w*dense_patch(patches[li][b], self.bank.patches[layer][ids[b]], True, 1)
                                for li, (layer, w) in enumerate([(3,.25),(6,.75)])))
        torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-6, atol=2e-7)

    def test_budget_adapts_chunks_without_dropping_references(self):
        bank = ArrayBank({3: torch.randn(30,64)}, {3: torch.randn(30,64,64)})
        query = torch.randn(64,64)
        retriever = ChunkedRetriever(bank, 'cpu', RetrievalConfig(30,64,4,1,1))
        ids = torch.arange(30)
        actual = retriever.patch_scores(3, query, ids, True, 2)
        expected = dense_patch(query, bank.patches[3], True, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)
        self.assertEqual(sum(bank.read_sizes), 30)
        self.assertLess(retriever.peak_reference_rows, 30)

    def test_shared_neighbors_are_read_once_per_layer_for_the_batch(self):
        patches = [torch.randn(4, 16, 12) for _ in (3, 6)]
        cls = [torch.randn(1, 12).repeat(4, 1) for _ in (3, 6)]
        retriever = ChunkedRetriever(self.bank, 'cpu', RetrievalConfig(2, 3, 4, 8, 8))
        actual, ids = retriever.score_batch(patches, cls, 7, True, 1)
        self.assertTrue(torch.equal(ids, ids[0].repeat(4, 1)))
        self.assertEqual(sum(self.bank.read_sizes), 7 * len(self.bank.layers))
        expected = torch.stack([sum(
            dense_patch(patches[li][b], self.bank.patches[layer][ids[b]], True, 1) / 2
            for li, layer in enumerate(self.bank.layers)) for b in range(4)])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)

    def test_invalid_requests_fail_before_bank_reads(self):
        retriever = ChunkedRetriever(self.bank, 'cpu')
        for k in (0, 14):
            with self.assertRaises(ValueError):
                retriever.image_neighbors(torch.randn(1,12), k)
        with self.assertRaises(ValueError):
            retriever.patch_scores(3, torch.randn(15,12), torch.tensor([0]))
        self.bank.n_samples = 400_000
        retriever = ChunkedRetriever(self.bank, 'cpu', RetrievalConfig(max_score_mib=1))
        with self.assertRaises(MemoryError):
            retriever.image_neighbors(torch.randn(1,12), 1)


if __name__ == '__main__':
    unittest.main()
