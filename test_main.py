import pytest
from test.utils import extract_solution
from subprocess import check_output


class TestMainFile:

    @pytest.fixture
    def run_main(self):
        def run_main_with_args(main_path, data_path, output_file):
            command = ["python3", main_path, data_path, output_file]
            result = check_output(command)
            return result.decode("utf-8")
        return run_main_with_args

    @pytest.mark.parametrize("main_path, output_file, template_file, train_dataset, min_percentage",
                             [("main.py", "output.txt", "train_dataset/bboxes_gt.txt", "train_dataset", 50.0)])
    def test_run_test(self, run_main, main_path, output_file, template_file, train_dataset, min_percentage):
        _ = run_main(main_path, train_dataset, output_file)

        with open(output_file, 'r') as f:
            output_content = f.read().strip()

        template_content = extract_solution(template_file)

        output_numbers = [list(map(int, line.split())) for line in output_content.split('\n')]
        template_numbers = [list(map(int, line.split())) for line in template_content.split('\n')]

        # Count the total number of numbers and the number of correct numbers
        total_numbers = sum(len(sublist) for sublist in template_numbers)
        correct_numbers = sum(1 for sublist_out, sublist_template in zip(output_numbers, template_numbers)
                              for num_out, num_template in zip(sublist_out, sublist_template) if num_out == num_template)

        percentage_correct = (correct_numbers / total_numbers) * 100

        assert percentage_correct > min_percentage