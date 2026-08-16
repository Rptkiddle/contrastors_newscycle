"""Key-id false-negative masking tests (multi-positive news training).

Run before training:  python -m pytest tests/test_key_masking.py -q
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from contrastors.loss import clip_loss, mask_false_negatives


def _make_batch():
    """2 queries, stride 2 (1 positive + 1 negative each) → 4 documents.

    Planted collision: q0's mined negative (document column 1) carries q1's
    key — it is a correct answer for q1 and must not be pushed away from it.
    """
    torch.manual_seed(0)
    query_keys = torch.tensor([11, 22], dtype=torch.int64)
    doc_keys = torch.tensor([11, 22, 22, 33], dtype=torch.int64)
    #                       q0pos q0neg q1pos q1neg
    labels = torch.tensor([0, 2])
    return query_keys, doc_keys, labels


def test_masks_collision_but_not_own_positive():
    query_keys, doc_keys, labels = _make_batch()
    sim = torch.zeros(2, 4)
    masked = mask_false_negatives(sim, query_keys, doc_keys, labels)
    lowest = torch.finfo(sim.dtype).min
    # q1 (key 22) vs q0's negative (key 22) is the planted collision
    assert masked[1, 1] == lowest
    # own positives untouched
    assert masked[0, 0] == 0 and masked[1, 2] == 0
    # everything else untouched
    assert masked[0, 1] == 0 and masked[0, 2] == 0 and masked[0, 3] == 0
    assert masked[1, 0] == 0 and masked[1, 3] == 0


def test_planted_collision_gets_zero_gradient():
    query_keys, doc_keys, labels = _make_batch()
    sim = torch.randn(2, 4, requires_grad=True)
    masked = mask_false_negatives(sim, query_keys, doc_keys, labels)
    loss = F.cross_entropy(masked, labels)
    loss.backward()
    # the masked collision cell contributes nothing to the loss
    assert sim.grad[1, 1].item() == 0.0
    # unmasked cells (positives and true negatives) all receive gradient
    for i, j in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3)]:
        assert sim.grad[i, j].item() != 0.0, f"cell ({i},{j}) unexpectedly zero"


def test_clip_loss_end_to_end_with_masking():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("gloo", rank=0, world_size=1)

    query_keys, doc_keys, labels = _make_batch()
    torch.manual_seed(1)
    query = F.normalize(torch.randn(2, 8), dim=-1)
    document = F.normalize(torch.randn(4, 8), dim=-1)
    identity_scale = lambda x: x  # noqa: E731

    unmasked = clip_loss(query, document, identity_scale)
    masked = clip_loss(query, document, identity_scale,
                       query_key_ids=query_keys, document_key_ids=doc_keys)

    expected = F.cross_entropy(
        mask_false_negatives(query @ document.T, query_keys, doc_keys, labels), labels)
    assert torch.allclose(masked, expected)
    assert not torch.allclose(masked, unmasked)


def test_masking_is_noop_without_collisions():
    query_keys = torch.tensor([11, 22], dtype=torch.int64)
    doc_keys = torch.tensor([11, 33, 22, 44], dtype=torch.int64)  # no collision
    labels = torch.tensor([0, 2])
    sim = torch.randn(2, 4)
    masked = mask_false_negatives(sim.clone(), query_keys, doc_keys, labels)
    assert torch.equal(masked, sim)
