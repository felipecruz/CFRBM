"""Prepare ordered float32 rating profiles once for both runner orientations."""

import numpy as np

class PreparedProfiles:
    """Cache dense host arrays; epochs only slice or gather existing rows.

    Column and profile order follow the supplied IDs and training-file order.
    Test-only profiles are zero rows. Historically evaluation inserted them
    into the runners' defaultdict, so they also train from epoch 1 onward.
    Keeping that detail preserves batch sizes and the model's random stream.
    Memory is O(profiles * visible IDs * 5), with two float32 matrices.
    """

    def __init__(self, profiles, visible_ids, tests):
        self.visible_index = {}
        for index, visible_id in enumerate(visible_ids):
            # Match list.index even if a caller supplies repeated IDs.
            self.visible_index.setdefault(visible_id, index)
        self.training_count = len(profiles)
        profile_ids = list(dict.fromkeys([*profiles, *tests]))
        self.profile_index = {pid: i for i, pid in enumerate(profile_ids)}
        self.test_ids = list(tests)
        self.test_rows = np.array([self.profile_index[pid] for pid in tests],
                                  dtype=np.intp)

        shape = (len(profile_ids), len(visible_ids))
        ratings = np.zeros(shape, dtype=np.float64)
        self.masks = np.zeros((shape[0], shape[1] * 5), dtype=np.float32)
        grouped_masks = self.masks.reshape(*shape, 5)
        for profile_id, entries in profiles.items():
            row = self.profile_index[profile_id]
            for visible_id, rating in entries:
                col = self.visible_index[visible_id]
                # Sequential assignment retains last-rating-wins semantics.
                ratings[row, col] = rating
                grouped_masks[row, col, :] = 1

        self.values = np.zeros_like(self.masks)
        rows, cols = np.nonzero(ratings)
        offsets = np.rint(ratings[rows, cols]).astype(np.intp) - 1
        self.values[rows, cols * 5 + offsets] = 1

    def training_batches(self, epoch, batch_size):
        stop = self.training_count if epoch == 0 else len(self.values)
        for start in range(0, stop, batch_size):
            end = min(start + batch_size, stop)
            yield self.values[start:end], self.masks[start:end]

    def test_batches(self, batch_size):
        for start in range(0, len(self.test_ids), batch_size):
            rows = self.test_rows[start:start + batch_size]
            yield self.test_ids[start:start + batch_size], self.values[rows]
