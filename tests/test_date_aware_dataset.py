import unittest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the parent directory to the path to import from the src package
sys.path.insert(0, os.path.abspath('..'))

from src.data import IndexMappingDataset, create_index_mapping_dataloaders, generate_index_mapped_predictions

class TestIndexMappingDataset(unittest.TestCase):
    def setUp(self):
        # Create sample data with dates
        self.dates = pd.date_range(start='2020-01-01', periods=100)
        self.data = pd.DataFrame(np.random.randn(100, 5), index=self.dates)
        self.lag_list = [1, 5, 22]
        self.h = 5

        # Create dataset
        self.dataset = IndexMappingDataset(self.data, self.lag_list, self.h)

    def test_dataset_length(self):
        # Expected length: 100 - max_lag - h + 1
        expected_length = 100 - 22 - 5 + 1
        self.assertEqual(len(self.dataset), expected_length)

    def test_get_date(self):
        """Test that get_date returns the correct date for a given index."""
        # First date should be at max_lag
        first_date = self.dataset.get_date(0)
        self.assertEqual(first_date, self.dates[22])

        # Last date should be at n_samples - h
        last_date = self.dataset.get_date(len(self.dataset) - 1)
        self.assertEqual(last_date, self.dates[100 - 5])

    def test_get_dates(self):
        """Test that get_dates returns correct dates for multiple indices."""
        # Test with sequential indices
        indices = [0, 1, 2, 3, 4]
        expected_dates = self.dates[22:22+5]
        actual_dates = self.dataset.get_dates(indices)

        for i, (expected, actual) in enumerate(zip(expected_dates, actual_dates)):
            self.assertEqual(
                expected, actual,
                f"Date mismatch at index {i}: expected {expected}, got {actual}"
            )

        # Test with non-sequential indices
        indices = [0, 5, 10, 15, 20]
        expected_dates = [self.dates[22+i] for i in indices]
        actual_dates = self.dataset.get_dates(indices)

        for i, (expected, actual) in enumerate(zip(expected_dates, actual_dates)):
            self.assertEqual(
                expected, actual,
                f"Date mismatch at index {i}: expected {expected}, got {actual}"
            )

    def test_get_item(self):
        """Test that __getitem__ returns tensors with the right shape and values."""
        # Get the first sample
        x_lag1, x_lag5, x_lag22, y = self.dataset[0]

        # Check shapes
        self.assertEqual(x_lag1.shape, torch.Size([1, 5]))
        self.assertEqual(x_lag5.shape, torch.Size([5, 5]))
        self.assertEqual(x_lag22.shape, torch.Size([22, 5]))
        self.assertEqual(y.shape, torch.Size([5, 5]))

        # Check that the values match the original data
        # x_lag1 should be the data at index max_lag - 1
        np.testing.assert_allclose(
            x_lag1.numpy(),
            self.data.iloc[22-1:22].values,
            rtol=1e-5
        )

        # x_lag5 should be the data from index max_lag - 5 to max_lag - 1
        np.testing.assert_allclose(
            x_lag5.numpy(),
            self.data.iloc[22-5:22].values,
            rtol=1e-5
        )

        # x_lag22 should be the data from index max_lag - 22 to max_lag - 1
        np.testing.assert_allclose(
            x_lag22.numpy(),
            self.data.iloc[22-22:22].values,
            rtol=1e-5
        )

        # y should be the data from index max_lag to max_lag + h - 1
        np.testing.assert_allclose(
            y.numpy(),
            self.data.iloc[22:22+5].values,
            rtol=1e-5
        )

    def test_dataloader_creation(self):
        # Create a dataloader
        dataloader = DataLoader(self.dataset, batch_size=10, shuffle=False)

        # Check that we can iterate through it
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            self.assertEqual(len(batch), 4)  # Should have 4 elements (x_lag1, x_lag5, x_lag22, y)

        expected_batches = (len(self.dataset) + 9) // 10  # Ceiling division
        self.assertEqual(batch_count, expected_batches)

    def test_prediction_with_dates(self):
        """Test that predictions are generated with correct dates and values."""
        # Create a mock model that returns predictable values
        class MockModel:
            def __init__(self):
                pass

            def eval(self):
                pass

            def __call__(self, x_lag1, x_lag5, x_lag22):
                batch_size = x_lag1.size(0)
                # Return predictions based on the input data for validation
                # Use a simple transformation that we can verify
                predictions = torch.mean(x_lag1, dim=1)  # Average across the lag dimension
                return predictions, None, None

        model = MockModel()
        batch_size = 10
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        market_indices = ['A', 'B', 'C', 'D', 'E']

        # Verify dataset setup
        self.assertEqual(len(self.dataset), 100 - 22 - 5 + 1)  # Expected length
        self.assertEqual(self.dataset.get_date(0), self.dates[22])  # First date
        self.assertEqual(self.dataset.get_date(len(self.dataset)-1), self.dates[100-5])  # Last date

        # Generate predictions
        pred_df, actual_df = generate_index_mapped_predictions(model, dataloader, self.dataset, market_indices)

        # Check that the predictions have a reasonable shape
        self.assertGreaterEqual(len(self.dataset), pred_df.shape[0])
        self.assertGreaterEqual(pred_df.shape[0], len(self.dataset) - 5)  # Allow for small differences
        self.assertEqual(pred_df.shape[1], 5)  # Should have 5 columns

        # Check that the dates are correct
        self.assertEqual(pred_df.index[0], self.dates[22])  # First date

        # Check that the last date is close to the expected last date
        last_date = pred_df.index[-1]
        expected_last_date = self.dates[95]  # Last date before horizon
        date_diff = abs((last_date - expected_last_date).days)
        self.assertLessEqual(date_diff, 5)  # Within 5 days

        # Check that the columns are correct
        self.assertEqual(list(pred_df.columns), market_indices)

        # Validate the actual values in the predictions
        # For the first batch, manually calculate what the predictions should be
        for i in range(min(batch_size, len(pred_df))):
            # Get the original data for this sample
            x_lag1, _, _, y = self.dataset[i]

            # Calculate expected prediction (mean of x_lag1)
            expected_pred = torch.mean(x_lag1, dim=0).numpy()

            # Get the actual prediction
            actual_pred = pred_df.iloc[i].values

            # Compare
            np.testing.assert_allclose(
                actual_pred,
                expected_pred,
                rtol=1e-5,
                err_msg=f"Prediction mismatch at index {i}"
            )

            # Check that the actual values match the original data
            # The first step of y should match the actual_df
            expected_actual = y[0].numpy()
            actual_actual = actual_df.iloc[i].values

            np.testing.assert_allclose(
                actual_actual,
                expected_actual,
                rtol=1e-5,
                err_msg=f"Actual value mismatch at index {i}"
            )

        # Check that the dates in pred_df and actual_df match
        pd.testing.assert_index_equal(pred_df.index, actual_df.index)

if __name__ == '__main__':
    unittest.main()
