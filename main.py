import sys
import math
import numpy as np
import pandas as pd
import cv2


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

    def run(self) -> dict | None:
        data = self.load_data()
        self.frames_following(data)
        return None

    def load_data(self) -> pd.DataFrame:
        data = []
        with open(self.bboxes_path, 'r') as file:
            lines = file.readlines()
        bb_count = 0
        row = []
        bboxes = []
        for line in lines:
            line = line.strip()
            # load img name
            if bb_count == 0:
                row = [line]
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

        df = pd.DataFrame(data, columns=['image_name', 'bounding_boxes'])
        return df

    def frames_following(self, df: pd.DataFrame):
        before_frame_data = None
        for index, row in df.iterrows():
            actual_frame_data = []
            for bbox in row['bounding_boxes']:
                if before_frame_data is not None:
                    for before_object_data in before_frame_data:
                        pass
            before_frame_data = actual_frame_data

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return image

    def calculate_center(self, bbox: list[int]) -> tuple[int, int]:
        x, y, w, h = bbox
        x_center = int(x + (w / 2))
        y_center = int(y + (h / 2))
        return x_center, y_center

    @staticmethod
    def calculate_distance(point1: tuple[int, int], point2: tuple[int, int]) -> float:
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def draw_image(self, img: np.ndarray, bboxes: list[list[int]], centers: list[tuple[int, int]]):
        image_with_bboxes = self.draw_bboxes(img, bboxes)
        image_with_centers = self.draw_centers(image_with_bboxes, centers)
        self.show_img(image_with_centers)

    @staticmethod
    def draw_bboxes(img: np.ndarray, bboxes: list[list[int]]) -> np.ndarray:
        output_img = img.copy()
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return output_img

    @staticmethod
    def draw_centers(img: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
        output_image = img.copy()
        for center in centers:
            cv2.circle(output_image, center, 2, (0, 255, 0), -1)
        return output_image

    @staticmethod
    def show_img(img: np.ndarray):
        cv2.imshow('img', img)
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
