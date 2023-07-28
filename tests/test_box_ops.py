import pytest 

import torch as th
from ml.vision.ops import enlarge_boxes, box_intersect

@pytest.mark.essential
def test_enlarge_boxes_boxes_no_enlarge_ratio():
    # Test case with no enlarge_ratio specified
    # Ensure the output boxes are enlarged to square in the longest side
    boxes = th.tensor([[1, 2, 5, 8]], dtype=th.float)  # [x1, y1, x2, y2]
    expected_output = th.tensor([[0, 2, 6, 8]], dtype=th.float)  # Enlarged to a square
    output = enlarge_boxes(boxes)
    assert th.allclose(output, expected_output)

@pytest.mark.essential
def test_enlarge_boxes_boxes_with_enlarge_ratio():
    # Test case with enlarge_ratio specified as a float
    boxes = th.tensor([[1, 2, 5, 8]], dtype=th.float)
    enlarge_ratio = 0.5  # Padding ratio
    expected_output = th.tensor([[-1.5, -1.5, 7.5, 11.5]], dtype=th.float)  # Padded with the given ratio
    output = enlarge_boxes(boxes, enlarge_ratio=enlarge_ratio)
    assert th.allclose(output, expected_output)

@pytest.mark.essential
def test_enlarge_boxes_boxes_with_enlarge_ratio_tuple():
    # Test case with enlarge_ratio specified as a tuple (width_enlarge_ratio, height_enlarge_ratio)
    boxes = th.tensor([[1, 2, 5, 8]], dtype=th.float)
    enlarge_ratio = (1.0, 2.0)  # Padding ratios for width and height respectively
    expected_output = th.tensor([[-4, -12, 10, 22]], dtype=th.float)  # Padded with different ratios for width and height
    output = enlarge_boxes(boxes, enlarge_ratio=enlarge_ratio)
    assert th.allclose(output, expected_output)

@pytest.mark.essential
def test_box_intersect():
    # Test case for intersection of boxes
    boxes1 = th.tensor([[1, 2, 5, 8], [2, 3, 6, 9]])
    boxes2 = th.tensor([[3, 4, 7, 10], [2, 4, 8, 11]])
    expected_output = th.tensor([[8, 12], [15, 20]])  # Intersection areas of boxes1 with boxes2
    output = box_intersect(boxes1, boxes2)
    assert th.allclose(output, expected_output)