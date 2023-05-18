import cv2
import sys


def get_dataset_path():
    if len(sys.argv) < 2:
        print("Add path to dataset as first argument!")
        sys.exit(1)
    file_path = sys.argv[1]
    return file_path


if __name__ == '__main__':
    path = get_dataset_path()
    print(path)
