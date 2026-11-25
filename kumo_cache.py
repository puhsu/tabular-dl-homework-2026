"""Encode context once; reuse column summaries and ICL memory for each query batch."""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class ContextCache:
    mean: Tensor
    columns: list[Tensor] = field(default_factory=list)
    memory: list[Tensor] = field(default_factory=list)


def embed_labels(module, labels):
    if hasattr(module, "y_emb"):
        return module.y_emb(labels.long())
    return module.y_lin(labels[:, None])


def encode_rows(model, x, categorical, cache, chunk, labels=None):
    encoder = model.row_embedding
    cells = x.new_empty(len(x), len(categorical), encoder.readout_token.size(-1))
    for output, batch in zip(cells.split(chunk), x.split(chunk), strict=True):
        output.copy_(encoder.cell_embedding(batch, 0, categorical, cache.mean))
    if labels is not None:
        cells += embed_labels(encoder, labels)[:, None]
    readout = encoder.readout_token.expand(len(x), -1, -1)
    for index, (column_block, row_block) in enumerate(
        zip(encoder.col_blocks, encoder.row_blocks, strict=True)
    ):
        if labels is not None:
            summaries = [
                column_block.inducing_block(column_block.inducing_points, column)
                for column in cells.transpose(0, 1)
            ]
            cache.columns.append(torch.stack(summaries))
        last = index == len(encoder.row_blocks) - 1
        next_readout = torch.empty_like(readout, memory_format=torch.contiguous_format)
        for start in range(0, len(x), chunk):
            rows = slice(start, start + chunk)
            columns = cells[rows].transpose(0, 1)
            updated = column_block.output_block(
                columns, cache.columns[index]
            ).transpose(0, 1)
            tokens = torch.cat((readout[rows], updated), dim=1)
            output = row_block(readout[rows] if last else tokens, tokens)
            next_readout[rows] = output[:, :4]
            if not last:
                # Safe to overwrite: all context summaries for this layer are ready.
                cells[rows] = output[:, 4:]
        readout = next_readout
    return model.row_project(encoder.norm(readout).flatten(1))


@torch.inference_mode()
def build_context_cache(model, x, labels, categorical, chunk):
    cache = ContextCache(x.nanmean(0).nan_to_num())
    rows = encode_rows(model, x, categorical, cache, chunk, labels)
    icl = model.icl_block
    rows += embed_labels(icl, labels)
    for index, layer in enumerate(icl.layers):
        cache.memory.append(rows)
        if index + 1 < len(icl.layers):
            rows = torch.cat([layer(batch, rows) for batch in rows.split(chunk)])
    return cache


@torch.inference_mode()
def predict_batch(model, x, categorical, cache, chunk):
    rows = encode_rows(model, x, categorical, cache, chunk)
    icl = model.icl_block
    for layer, memory in zip(icl.layers, cache.memory, strict=True):
        rows = layer(rows, memory, kv_heads=2)
    return icl.head(icl.norm(rows))
