import sys
import math
import numpy as np
import pandas as pd
import cv2


class PersonTracker:
    def __init__(self, files_path: str, bboxes_path: str = None, images_path: str = None):
        self.files_path = files_path or get_dataset_path_as_arg()
        self.bboxes_path = bboxes_path or f"{files_path}/bboxes.txt"
        self.images_path = images_path or f"{files_path}/frames/"

    def run(self) -> dict | None:
        data = self.load_data()
        self.frames_following(data)
        return None

    def load_data(self) -> pd.DataFrame:
        data = []

        with open(self.bboxes_path, 'r') as file:
            lines = file.readlines()

        bboxes_count = 0
        row = []
        bboxes = []

        for line in lines:

            line = line.strip()

            # load img name
            if bboxes_count == 0:
                # add image_name to row
                row = [line]
                bboxes = []
                bboxes_count = -1

            # load number of bboxes
            elif bboxes_count == -1:
                bboxes_count = int(line)

            # load bboxes
            else:
                # convert bbox string to list of floats
                bbox_points = [int(float(number)) for number in line.split()]
                bboxes.append(bbox_points)
                bboxes_count -= 1

                # if all bboxes loaded, append row to data
                if bboxes_count == 0:
                    row.append(bboxes)
                    data.append(row)

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
    def load_image(img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        return img

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

    def cut_image(self, image, bbox):
        x, y, w, h = bbox
        cut_image = image[y:y + h, x:x + w]
        return cut_image

    def draw_image(self, img: np.ndarray, bboxes: list[list[int]], centers: list[tuple[int, int]]):
        img_with_bboxes = self.draw_bboxes(img, bboxes)
        img_with_centers = self.draw_centers(img_with_bboxes, centers)
        self.show_img(img_with_centers)

    @staticmethod
    def draw_bboxes(img: np.ndarray, bboxes: list[list[int]]) -> np.ndarray:
        output_img = img.copy()
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return output_img

    @staticmethod
    def draw_centers(img: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
        output_img = img.copy()
        for center in centers:
            cv2.circle(output_img, center, 2, (0, 0, 255), -1)
        return output_img

    @staticmethod
    def show_img(img: np.ndarray):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_dataset_path_as_arg() -> str:
    if len(sys.argv) < 2:
        print("Add files_path to dataset as first argument!")
        sys.exit(1)
    file_path = sys.argv[1]
    return file_path


if __name__ == '__main__':
    tracker = PersonTracker(get_dataset_path_as_arg())
    print(tracker.run())
