from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler
from typing import Iterator, List, Sequence, Set, Tuple

def _scalar_int(value, name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required graph property {name!r}.")

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must contain one graph-level integer, "
                f"got shape {tuple(value.shape)}."
            )
        return int(value.reshape(-1)[0].item())

    return int(value)


class SameGeometryBatchSampler(Sampler[List[int]]):
    """Keep several method labels of one exact geometry in one batch.

    The sampler is compatible with single-process and distributed training.
    In DDP, complete batches are assigned to ranks, so a pair group is never
    split between ranks.
    """

    def __init__(
        self,
        dataset: Sequence,
        batch_size: int,
        methods_per_structure: int = 4,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if methods_per_structure < 2:
            raise ValueError(
                "methods_per_structure must be at least 2."
            )

        if methods_per_structure > batch_size:
            raise ValueError(
                "methods_per_structure cannot exceed batch_size."
            )

        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")

        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank={rank} is invalid for num_replicas={num_replicas}."
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.methods_per_structure = int(methods_per_structure)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

        groups = defaultdict(list)
        methods_by_structure = defaultdict(set)

        for dataset_index in range(len(dataset)):
            item = dataset[dataset_index]

            structure_index = _scalar_int(
                getattr(item, "structure_index", None),
                "structure_index",
            )

            method_index = _scalar_int(
                getattr(item, "method_index", None),
                "method_index",
            )

            groups[structure_index].append(dataset_index)
            methods_by_structure[structure_index].add(method_index)

        self.groups = dict(groups)
        self.methods_by_structure = {
            structure_index: set(method_indices)
            for structure_index, method_indices
            in methods_by_structure.items()
        }

        self.num_structures = len(self.groups)

        # A singleton has only one distinct method, even if that frame was
        # accidentally duplicated in the input dataset.
        self.num_singletons = sum(
            len(method_indices) == 1
            for method_indices in self.methods_by_structure.values()
        )

        # A geometry is genuinely pairable only if at least two distinct
        # methods are available.
        self.num_pairable = sum(
            len(method_indices) >= 2
            for method_indices in self.methods_by_structure.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_global_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        chunks: List[Tuple[int, List[int]]] = []

        for structure_index, indices in self.groups.items():
            local_indices = list(indices)

            if self.shuffle:
                rng.shuffle(local_indices)

            for start in range(
                0,
                len(local_indices),
                self.methods_per_structure,
            ):
                chunks.append(
                    (
                        structure_index,
                        local_indices[
                            start : start + self.methods_per_structure
                        ],
                    )
                )

        if self.shuffle:
            rng.shuffle(chunks)

        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_structure_indices: Set[int] = set()

        for structure_index, chunk in chunks:
            would_overflow = (
                len(current_batch) + len(chunk) > self.batch_size
            )

            duplicate_structure = (
                structure_index in current_structure_indices
            )

            if current_batch and (
                would_overflow or duplicate_structure
            ):
                batches.append(current_batch)
                current_batch = []
                current_structure_indices = set()

            current_batch.extend(chunk)
            current_structure_indices.add(structure_index)

        if current_batch:
            batches.append(current_batch)

        if len(batches) == 0:
            return []

        # DDP needs the same number of optimizer steps on every rank.
        remainder = len(batches) % self.num_replicas

        if remainder:
            number_to_add = self.num_replicas - remainder
            batches.extend(
                list(batches[i % len(batches)])
                for i in range(number_to_add)
            )

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        global_batches = self._build_global_batches()
        local_batches = global_batches[
            self.rank :: self.num_replicas
        ]
        return iter(local_batches)

    def __len__(self) -> int:
        global_batches = self._build_global_batches()
        return len(global_batches) // self.num_replicas