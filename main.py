import cv2
import sys

import numpy as np


class PersonTracker:
    def __init__(self, path: str, bbox_path: str = None, frames_path: str = None):

        if path is None:
            path = get_dataset_path_as_arg()
        if bbox_path is None:
            bbox_path = path + "/bboxes.txt"
        if frames_path is None:
            frames_path = path + "/frames/"

        self.files_path = path
        self.bboxes_path = bbox_path
        self.frames_path = frames_path

    def run(self) -> dict|None:
        data = self.load_data()
        data_with_center = self.find_centers(data)
        self.draw_images(data_with_center)
        return None

    def load_data(self) -> np.ndarray:
        data = []
        with open(self.bboxes_path, 'r') as file:
            lines = file.readlines()
        bb_count = 0
        row = []
        bboxes = []
        for line in lines:
            line = line.strip()
            # load image name
            if bb_count == 0:
                image = self.load_image(self.frames_path + line)
                row = [line, image]
                bboxes = []
                bb_count = -1
            # load number of bboxes
            elif bb_count == -1:
                bb_count = int(line)
            # load bboxes
            else:
                # convert bbox string to list of floats
                bbox_points = [int(float(number)) for number in line.split()]
                bboxes.append(bbox_points)
                if bb_count == 1:
                    row.append(bboxes)
                    data.append(row)
                bb_count -= 1
        return np.array(data, dtype=object)

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return image

    def find_centers(self, data: np.ndarray) -> np.ndarray:
        centers_column = np.empty((data.shape[0], 1), dtype=object)
        for i, row in enumerate(data):
            centers = []
            for bbox in row[2]:
                centers.append(self.find_center(bbox))
            centers_column[i, 0] = centers
        data_with_centers = np.hstack((data, centers_column))
        return data_with_centers

    def find_center(self, bbox: list[int]) -> tuple[int, int]:
        x, y, w, h = bbox
        x_center = int(x + (w / 2))
        y_center = int(y + (h / 2))
        return x_center, y_center

    def draw_images(self, data: np.ndarray):
        for row in data:
            image_with_bboxes = self.draw_bboxes(row[1], row[2])
            image_with_centers = self.draw_centers(image_with_bboxes, row[3])
            self.show_image(image_with_centers)

    def draw_bboxes(self, image: np.ndarray, bboxes: list[list[int]]) -> np.ndarray:
        output_image = image.copy()
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return output_image

    def draw_centers(self, image: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
        output_image = image.copy()
        for center in centers:
            cv2.circle(output_image, center, 2, (0, 255, 0), -1)
        return output_image

    def show_image(self, image: np.ndarray):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_dataset_path_as_arg() -> str:
    if len(sys.argv) < 2:
        print("Add path to dataset as first argument!")
        sys.exit(1)
    file_path = sys.argv[1]
    return file_path


if __name__ == '__main__':
    tracker = PersonTracker(get_dataset_path_as_arg())
    print(tracker.run())
