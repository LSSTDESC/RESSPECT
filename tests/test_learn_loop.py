import os
import pytest
import tempfile

from resspect import fit_snpcc
from resspect.learn_loop import learn_loop

def test_can_run_learn_loop(test_data_path):
    """Test that learn_loop can load data and run."""
    with tempfile.TemporaryDirectory() as dir_name:
        # Create the feature files to use for the learning loop.
        output_file = os.path.join(dir_name, "output_file.dat")
        fit_snpcc(
            path_to_data_dir=test_data_path,
            features_file=output_file,
        )

        learn_loop(
            nloops=1,
            features_method="bazin",
            strategy="RandomSampling",
            path_to_features=output_file,
            output_metrics_file=os.path.join(dir_name,"just_a_name.csv"),
            output_queried_file=os.path.join(dir_name,"just_other_name.csv"),
        )

if __name__ == '__main__':
    pytest.main()
