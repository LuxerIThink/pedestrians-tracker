from test import utils
import subprocess


class TestMain:

    def test_main(self):
        path_to_dataset = 'test/dataset'
        process = subprocess.run(['python', 'main.py', path_to_dataset], stdout=subprocess.PIPE)
        correct_result = utils.extract_solution(f'{path_to_dataset}/bboxes_gt.txt')
        assert process.returncode == correct_result