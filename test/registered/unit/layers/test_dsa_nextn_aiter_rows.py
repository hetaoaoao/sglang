from pathlib import Path


DSA_BACKEND = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "dsa_backend.py"
)


def test_nextn_aiter_decode_scratch_scales_with_draft_tokens():
    max_bs = 32
    draft_tokens = 4
    dsa_index_topk = 2048

    old_indptr_capacity = max_bs + 1
    old_indices_capacity = max_bs * dsa_index_topk

    attn_rows = max_bs * draft_tokens
    required_indptr_capacity = attn_rows + 1
    required_indices_capacity = attn_rows * dsa_index_topk

    assert old_indptr_capacity < required_indptr_capacity
    assert old_indices_capacity < required_indices_capacity


def test_nextn_aiter_indptr_uses_attention_rows_not_request_batch():
    request_bs = 2
    draft_tokens = 4
    attn_rows = request_bs * draft_tokens
    page_table = [
        [11, 12, -1, -1],
        [21, -1, -1, -1],
        [31, 32, 33, -1],
        [41, 42, 43, 44],
        [51, -1, -1, -1],
        [61, 62, -1, -1],
        [71, 72, 73, -1],
        [81, 82, 83, 84],
    ]
    non_minus1_counts = [sum(value != -1 for value in row) for row in page_table]

    assert len(non_minus1_counts) == attn_rows

    indptr = [0]
    for count in non_minus1_counts:
        indptr.append(indptr[-1] + count)

    assert len(indptr) == attn_rows + 1
    assert indptr[request_bs] != indptr[attn_rows]


def test_source_uses_attention_rows_for_aiter_nextn_decode():
    text = DSA_BACKEND.read_text()

    assert (
        "max_decode_rows = max_bs * max(self.speculative_num_draft_tokens or 1, 1)"
        in text
    )
    assert "(max_decode_rows + 1,)" in text
    assert "max_decode_rows * self.dsa_index_topk" in text
    assert "attn_rows = page_table_1.shape[0]" in text
    assert (
        "kv_indptr[1 : attn_rows + 1] = "
        "torch.cumsum(non_minus1_counts, dim=0)"
    ) in text
    assert (
        "get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, attn_rows)"
        in text
    )

    assert (
        "kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)"
        not in text
    )
    assert "get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)" not in text
    assert "kv_indptr[0] = 0" not in text
