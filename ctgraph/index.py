"""Spatial index over a label volume: point / slice / slab / box -> structures.

All slice and slab queries are answered from a per-slice voxel-count table and its
prefix sum along z, so they are O(K) regardless of volume size.
"""
from __future__ import annotations

import numpy as np

from .volume import Case


class SpatialIndex:
    def __init__(self, case: Case):
        self.case = case
        K = len(case.names) + 1
        lab = case.labels
        # counts[z, k] = number of voxels of label k on axial slice z
        self.counts = np.stack([np.bincount(lab[:, :, z].ravel(), minlength=K) for z in range(lab.shape[2])])
        self.cum = self.counts.cumsum(0)
        self.pixel_area_mm2 = case.spacing[0] * case.spacing[1]

    def _ids_to_names(self, ids) -> list[str]:
        return [self.case.names[i - 1] for i in ids if i > 0]

    def point(self, x: int, y: int, z: int) -> list[str]:
        return self._ids_to_names([int(self.case.labels[x, y, z])])

    def slice_nodes(self, z: int) -> list[str]:
        return self._ids_to_names(np.nonzero(self.counts[z])[0])

    def slice_areas_mm2(self, z: int) -> dict[str, float]:
        ids = np.nonzero(self.counts[z])[0]
        return {self.case.names[i - 1]: float(self.counts[z, i] * self.pixel_area_mm2) for i in ids if i > 0}

    def slab_nodes(self, z0: int, z1: int) -> list[str]:
        """Structures present anywhere in slices z0..z1 (inclusive)."""
        z0, z1 = max(z0, 0), min(z1, self.counts.shape[0] - 1)
        if z0 > z1:
            return []
        tot = self.cum[z1] - (self.cum[z0 - 1] if z0 > 0 else 0)
        return self._ids_to_names(np.nonzero(tot)[0])

    def above(self, z: int) -> list[str]:
        """Superior to slice z (+z is superior in the RAS frame)."""
        return self.slab_nodes(z + 1, self.counts.shape[0] - 1)

    def below(self, z: int) -> list[str]:
        return self.slab_nodes(0, z - 1)

    def box_nodes(self, x0, x1, y0, y1, z0, z1) -> dict[str, int]:
        """Voxel counts of every structure inside an axis-aligned box (inclusive bounds)."""
        sub = self.case.labels[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]
        bc = np.bincount(sub.ravel(), minlength=len(self.case.names) + 1)
        return {self.case.names[i - 1]: int(bc[i]) for i in np.nonzero(bc)[0] if i > 0}

    def largest_slice(self, name: str) -> int:
        """Slice with the largest cross-section of a structure (a natural representative slice)."""
        return int(self.counts[:, self.case.name_to_id(name)].argmax())

    def z_range(self, name: str) -> tuple[int, int]:
        nz = np.nonzero(self.counts[:, self.case.name_to_id(name)])[0]
        return int(nz[0]), int(nz[-1])
