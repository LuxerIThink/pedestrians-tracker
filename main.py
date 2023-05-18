import cv2
import sys


class PersonTracker:
    def __init__(self, path: str, bbox_path: str = None, frames_path: str = None):

        if path is None:
            path = get_dataset_path_as_arg()
        if bbox_path is None:
            bbox_path = path + "/bboxes.txt"
        if frames_path is None:
            frames_path = path + "/frames"

        self.files_path = path
        self.bboxes_path = bbox_path
        self.frames_path = frames_path

    def run(self) -> dict:
        data = self.load_data()
        return data

    def load_data(self):
        data = {}
        current_key = None
        with open(self.bboxes_path, 'r') as file:
            lines = file.readlines()
            bb_count = 0
            for line in lines:
                line = line.strip()
                if bb_count == 0:
                    current_key = line
                    data[current_key] = []
                    bb_count = -1
                elif bb_count == -1:
                    bb_count = int(line)
                else:
                    data[current_key].append(line)
                    bb_count -= 1
        return data


def get_dataset_path_as_arg() -> str:
    if len(sys.argv) < 2:
        print("Add path to dataset as first argument!")
        sys.exit(1)
    file_path = sys.argv[1]
    return file_path


if __name__ == '__main__':
    tracker = PersonTracker(get_dataset_path_as_arg())
    print(tracker.run())
