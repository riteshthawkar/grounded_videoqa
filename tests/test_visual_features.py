import numpy as np

from adaptive_evidence_vqa.retrieval.visual_features import (
    aggregate_segment_embeddings,
    clip_output_to_numpy,
    select_segment_frame_indices,
)


class DummyTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class DummyOutput:
    def __init__(self, tensor):
        self.pooler_output = tensor


def test_select_segment_frame_indices_uses_overlap_when_available() -> None:
    frame_times = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    indices = select_segment_frame_indices(frame_times, start_time=0.5, end_time=2.5)

    assert indices.tolist() == [1, 2]


def test_select_segment_frame_indices_falls_back_to_nearest_frame() -> None:
    frame_times = np.asarray([0.0, 2.0, 4.0], dtype=np.float32)

    indices = select_segment_frame_indices(frame_times, start_time=2.6, end_time=3.0)

    assert indices.tolist() == [1]


def test_aggregate_segment_embeddings_averages_over_selected_frames() -> None:
    frame_times = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    frame_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    segments = [
        {"start": 0.0, "end": 1.1},
        {"start": 2.1, "end": 2.3},
    ]

    aggregated = aggregate_segment_embeddings(frame_times, frame_embeddings, segments)

    assert aggregated.shape == (2, 2)
    assert np.allclose(np.linalg.norm(aggregated, axis=1), 1.0)


def test_clip_output_to_numpy_uses_pooler_output() -> None:
    output = DummyOutput(DummyTensor([[1.0, 2.0, 3.0]]))

    array = clip_output_to_numpy(output)

    assert np.array_equal(array, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
