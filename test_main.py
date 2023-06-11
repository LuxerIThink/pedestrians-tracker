import pytest
from subprocess import check_output


class TestMainFile:

    @staticmethod
    def load_text_file(file_path: str) -> list:
        try:
            with open(file_path, 'r'):
                output = [[int(num) for num in line.split()] for line in open(file_path, 'r').read().splitlines()]
        except FileNotFoundError:
            raise FileNotFoundError(f'File {file_path} not found')
        return output

    @pytest.fixture
    def run_main(self):
        def run_main_with_args(main_path, data_path, output_file):
            command = ["python3", main_path, data_path, output_file]
            result = check_output(command)
            return result.decode("utf-8")
        return run_main_with_args

    @pytest.mark.parametrize("main_path, output_file, template_file, train_dataset, min_percentage",
                             [("main.py", "output.txt", "processing/solution.txt", "train_dataset", 80.0)])
    def test_main(self, run_main, main_path, output_file, template_file, train_dataset, min_percentage):
        # Run main.py
        _ = run_main(main_path, train_dataset, output_file)

        # Load output and template files
        template_numbers = self.load_text_file(template_file)
        output_numbers = self.load_text_file(output_file)

        # Count the total number of numbers and the number of correct numbers
        total_numbers = sum(len(sublist) for sublist in template_numbers)
        correct_numbers = sum(1 for sublist_out, sublist_template in zip(output_numbers, template_numbers)
                              for num_out, num_template in zip(sublist_out, sublist_template)
                              if num_out == num_template)

        percentage_correct = (correct_numbers / total_numbers) * 100

        # print(percentage_correct)
        assert percentage_correct >= min_percentage
