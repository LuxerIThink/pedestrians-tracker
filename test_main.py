import pytest
import numpy as np
from test.utils import load_text_file
from subprocess import check_output
from main import PersonTracker


class TestMainFile:

    @pytest.fixture
    def run_main(self):
        def run_main_with_args(main_path, data_path, output_file):
            command = ["python3", main_path, data_path, output_file]
            result = check_output(command)
            return result.decode("utf-8")
        return run_main_with_args

    @pytest.mark.parametrize("main_path, output_file, template_file, train_dataset, min_percentage",
                             [("main.py", "output.txt", "test/solution.txt", "train_dataset", 85.0)])
    def test_run_test(self, run_main, main_path, output_file, template_file, train_dataset, min_percentage):
        _ = run_main(main_path, train_dataset, output_file)

        template_numbers = load_text_file(template_file)
        output_numbers = load_text_file(output_file)

        # Count the total number of numbers and the number of correct numbers
        total_numbers = sum(len(sublist) for sublist in template_numbers)
        correct_numbers = sum(1 for sublist_out, sublist_template in zip(output_numbers, template_numbers)
                              for num_out, num_template in zip(sublist_out, sublist_template)
                              if num_out == num_template)

        percentage_correct = (correct_numbers / total_numbers) * 100

        # print(percentage_correct)
        assert percentage_correct >= min_percentage
