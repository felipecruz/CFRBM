import unittest

import numpy as np

import cfrbm.utils as utils

class UtilsTest(unittest.TestCase):
    def test_chunker(self):
        data = np.array(range(1, 101)).reshape(10, 10)

        for chunk in utils.chunker(data, 10):
            self.assertEqual(10, len(chunk))

    def test_expand_line(self):
        original = [3, 0, 0, 2]
        expected = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

        self.assertEqual(expected, utils._expand_line(original, k=3))

    def test_expand(self):
        original = np.array([
            [3, 0, 0, 5],
            [2, 1, 3, 0],
            [5, 4, 0, 3],
            [0, 0, 4, 0],
            [0, 2, 0, 2],
        ])

        expected = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])

        self.assertTrue(np.array_equal(expected, utils.expand(original)))

    def test_revert_expected_value(self):
        v1 = np.array([[1, 0, 0, 0, 0]])
        v2 = np.array([[0, 1, 0, 0, 0]])
        v3 = np.array([[0, 0, 1, 0, 0]])
        v4 = np.array([[0, 0, 0, 1, 0]])
        v5 = np.array([[0, 0, 0, 0, 1]])

        v42 = np.array([[.2, 0, 0, 0, .8]])

        self.assertEqual(1, utils.revert_expected_value(v1))
        self.assertEqual(2, utils.revert_expected_value(v2))
        self.assertEqual(3, utils.revert_expected_value(v3))
        self.assertEqual(4, utils.revert_expected_value(v4))
        self.assertEqual(5, utils.revert_expected_value(v5))
        self.assertEqual(4.2, utils.revert_expected_value(v42, do_round=False))

