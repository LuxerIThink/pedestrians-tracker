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
            frames_path = path + "/df/"

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
            # load image name
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

    @staticmethod
    def frames_following(df):
        before_frame_data = None
        for index, row in df.iterrows():
            actual_frame_data = []
            for bbox in row['bounding_boxes']:
                one_object_data = []
                if before_frame_data is not None:
                    pass
                actual_frame_data.append(one_object_data)
            print('\n')
            print(before_frame_data)
            print(actual_frame_data)
            before_frame_data = actual_frame_data

    def compare_frames(self, actual_frame, before_frame):
        pass

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return image

    @staticmethod
    def find_center(bbox: list[int]) -> tuple[int, int]:
        x, y, w, h = bbox
        x_center = int(x + (w / 2))
        y_center = int(y + (h / 2))
        return x_center, y_center

    def draw_images(self, data: np.ndarray):
        for row in data:
            image_with_bboxes = self.draw_bboxes(row[1], row[2])
            image_with_centers = self.draw_centers(image_with_bboxes, row[3])
            self.show_image(image_with_centers)

    @staticmethod
    def draw_bboxes(image: np.ndarray, bboxes: list[list[int]]) -> np.ndarray:
        output_image = image.copy()
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return output_image

    @staticmethod
    def draw_centers(image: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
        output_image = image.copy()
        for center in centers:
            cv2.circle(output_image, center, 2, (0, 255, 0), -1)
        return output_image

    @staticmethod
    def show_image(image: np.ndarray):
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
