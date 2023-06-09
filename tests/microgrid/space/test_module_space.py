import numpy as np

from pymgrid.utils.space import ModuleSpace
from tests.helpers.test_case import TestCase


class TestModuleSpace(TestCase):

    @staticmethod
    def get_space(unnormalized_low, unnormalized_high, normalized_bounds=(0, 1)):
        return ModuleSpace(unnormalized_low=unnormalized_low,
                           unnormalized_high=unnormalized_high,
                           normalized_bounds=normalized_bounds)

    def test_normalize(self):
        unnorm_low = np.zeros(2)
        unnorm_high = np.array([1, 2])

        space = self.get_space(unnorm_low, unnorm_high)

        vals_to_normalize = [np.zeros(2), np.array([0.5, 1]), np.array([1, 2])]
        expected_normalized_vals = [np.zeros(2), np.array([0.5, 0.5]), np.array([1, 1])]

        for val_to_normalize, expected_normalized_val in zip(vals_to_normalize, expected_normalized_vals):
            with self.subTest(val_to_normalize=val_to_normalize, expected_normalized_val=expected_normalized_val):
                normalized = space.normalize(val_to_normalize)
                self.assertEqual(normalized, expected_normalized_val)

    def test_denormalize(self):
        unnorm_low = np.zeros(2)
        unnorm_high = np.array([1, 2])

        space = self.get_space(unnorm_low, unnorm_high)

        vals_to_denormalize = [np.zeros(2), np.array([0.5, 0.5]), np.array([1, 1])]
        expected_denormalized_vals = [np.zeros(2), np.array([0.5, 1]), np.array([1, 2])]

        for val_to_denormalize, expected_denormalized_val in zip(vals_to_denormalize, expected_denormalized_vals):
            with self.subTest(val_to_denormalize=val_to_denormalize, expected_denormalized_val=expected_denormalized_val):
                denormalized = space.denormalize(val_to_denormalize)
                self.assertEqual(denormalized, expected_denormalized_val)

    def test_normalize_different_normalized_bounds(self):
        unnorm_low = np.zeros(2)
        unnorm_high = np.array([1, 2])
        normalized_bounds = [-3, 2]

        space = self.get_space(unnorm_low, unnorm_high, normalized_bounds=normalized_bounds)

        vals_to_normalize = [np.zeros(2), np.array([0.5, 1]), np.array([1, 2])]
        expected_normalized_vals = [np.array([-3, -3]), np.array([-0.5, -0.5]), np.array([2, 2])]

        for val_to_normalize, expected_normalized_val in zip(vals_to_normalize, expected_normalized_vals):
            with self.subTest(val_to_normalize=val_to_normalize, expected_normalized_val=expected_normalized_val):
                normalized = space.normalize(val_to_normalize)
                self.assertEqual(normalized, expected_normalized_val)

    def test_denormalize_different_normalized_bounds(self):
        unnorm_low = np.zeros(2)
        unnorm_high = 2 * np.arange(2)
        normalized_bounds = [-3, 2]

        space = self.get_space(unnorm_low, unnorm_high, normalized_bounds=normalized_bounds)

        vals_to_denormalize = [np.array([-3, -3]), np.array([-0.5, -0.5]), np.array([2, 2])]
        expected_denormalized_vals = [np.zeros(2), np.array([0.5, 1]), np.array([1, 2])]

        for val_to_denormalize, expected_denormalized_val in zip(vals_to_denormalize, expected_denormalized_vals):
            with self.subTest(val_to_denormalize=val_to_denormalize, expected_denormalized_val=expected_denormalized_val):
                denormalized = space.denormalize(val_to_denormalize)
                self.assertEqual(denormalized, expected_denormalized_val)

    def test_clip_no_clip_normalized(self):
        unnorm_low = np.zeros(2)
        unnorm_high = 2 * np.arange(2)
        normalized_bounds = [0, 2]

        space = self.get_space(unnorm_low, unnorm_high, normalized_bounds=normalized_bounds)

        val = (normalized_bounds[1] - normalized_bounds[0]) * np.random.rand(2) + normalized_bounds[0]

        self.assertIn(val, space.normalized)
        self.assertEqual(val, space.clip(val, normalized=True))

    def test_clip_yes_clip_normalized(self):
        unnorm_low = np.zeros(2)
        unnorm_high = 2 * np.arange(2)
        normalized_bounds = [0, 2]

        space = self.get_space(unnorm_low, unnorm_high, normalized_bounds=normalized_bounds)

        val = (normalized_bounds[1] - normalized_bounds[0]) * np.random.rand(2) + normalized_bounds[1]

        clipped = space.clip(val, normalized=True)

        self.assertNotIn(val, space.normalized)
        self.assertNotEqual(val, clipped)
        self.assertEqual(clipped, space.normalized.high)

    def test_clip_no_clip_unnormalized(self):
        unnorm_low = np.zeros(2)
        unnorm_high = 2 * np.arange(2)
        normalized_bounds = [0, 2]

        space = self.get_space(unnorm_low, unnorm_high, normalized_bounds=normalized_bounds)

        val = (unnorm_high - unnorm_low) * np.random.rand(2) + unnorm_low

        self.assertIn(val, space.unnormalized)
        self.assertEqual(val, space.clip(val, normalized=False))

    def test_clip_yes_clip_unnormalized(self):
        unnorm_low = np.zeros(2)
        unnorm_high = 2 * np.arange(2)
        normalized_bounds = [0, 2]

        space = self.get_space(unnorm_low, unnorm_high, normalized_bounds=normalized_bounds)

        val = (unnorm_high - unnorm_low) * np.random.rand(2) + unnorm_high

        clipped = space.clip(val, normalized=False)

        self.assertNotIn(val, space.unnormalized)
        self.assertNotEqual(val, clipped)
        self.assertEqual(clipped, space.unnormalized.high)
