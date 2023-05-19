import cv2
import sys

import numpy


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

    def run(self) -> dict:
        data = self.load_data()
        self.edit_images(data)
        return 0

    def load_data(self) -> dict:
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

    def edit_images(self, data: dict):
        for image_name, bounding_boxes in data.items():
            image = cv2.imread(self.frames_path + image_name)
            bbox_coords = [[int(float(x)) for x in bounding_box.split()] for bounding_box in bounding_boxes]
            self.edit_image(image, bbox_coords)

    def edit_image(self, image: numpy.ndarray, bboxes: list[list[int]]):
        image_with_bboxes = self.draw_bboxes(image, bboxes)
        image_with_bbox_centers = self.draw_centers(image_with_bboxes, bboxes)
        self.show_image(image_with_bbox_centers)

    def draw_bboxes(self, image: numpy.ndarray, bboxes: list[list[int]]) -> numpy.ndarray:
        output_image = image.copy()
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return output_image

    def draw_centers(self, image: numpy.ndarray, bboxes: list[list[int]]) -> numpy.ndarray:
        output_image = image.copy()
        centers = self.centers_of_the_boxes(bboxes)
        for center in centers:
            cv2.circle(output_image, center, 2, (0, 255, 0), -1)
        return output_image

    def centers_of_the_boxes(self, bboxes: list[list[int]]) -> list[list[int, int]]:
        centers = []
        for bbox in bboxes:
            centers.append(self.center_of_bbox(bbox))
        return centers

    def center_of_bbox(self, bbox: list[int]) -> tuple[int, int]:
        x, y, w, h = bbox
        x_center = int(x + (w / 2))
        y_center = int(y + (h / 2))
        return x_center, y_center

    def show_image(self, image: numpy.ndarray):
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
