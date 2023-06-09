import sys
import math
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment


class PersonTracker:
    def __init__(self, files_path: str = None, bboxes_path: str = None, images_path: str = None):
        self.files_path = files_path or get_dataset_path_as_arg()
        self.bboxes_path = bboxes_path or f"{files_path}/bboxes.txt"
        self.images_path = images_path or f"{files_path}/frames/"
        self.image_shape = None
        self.min_prob = 0.13

    def run(self) -> dict | str:
        data = self.load_data()
        self.set_image_shape(data)
        frame_numbers = self.frames_following(data)
        outputs = self.convert_list(frame_numbers)
        return outputs

    def set_image_shape(self, data: pd.DataFrame):
        image = cv2.imread(f"{self.images_path}/{data.iloc[0]['name']}")
        self.image_shape = image.shape

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

        df = pd.DataFrame(data, columns=['name', 'bounding_boxes'])

        return df

    def frames_following(self, df):
        frames_numbers = []
        frame_data = None
        frame_numbers = []
        for _, row in df.iterrows():
            frame_numbers, frame_data = self.get_image_frames(row, frame_data, frame_numbers)
            frames_numbers.append(list(frame_numbers))
        return frames_numbers

    def get_image_frames(self, image_data, before_data, before_numbers):
        actual_data = self.gather_features(image_data)
        probability_matrix = []
        if before_data is not None:
            for before_object in before_data:
                probability_for_before_object = []
                for actual_object in actual_data:
                    probability1 = self.template_matching(before_object['image'],
                                                          actual_object['image'])
                    probability2 = self.histogram_similarity(before_object['histogram'],
                                                             actual_object['histogram'])
                    probability_for_before_object.append(probability1 * probability2)
                probability_matrix.append(probability_for_before_object)
            probability_matrix = np.array(probability_matrix)

            actual_numbers = self.fit_objects(before_numbers, probability_matrix)
            print(f'actual numbers: {actual_numbers}')
        else:
            actual_numbers = np.array([-1 for _ in range(len(actual_data))])
        return actual_numbers, actual_data

    # compare two images with Template Matching
    @staticmethod
    def template_matching(image1, image2):
        # Resize the images to a consistent size for comparison
        size = (128, 128)
        image1 = cv2.resize(image1, size)
        image2 = cv2.resize(image2, size)
        # Apply template Matching
        res = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val

    def fit_objects(self, previous_indexes: list[int], probability_matrix: np.ndarray[float]):

        # if value is lower than min_prob, set it to 0
        probability_matrix = np.where(probability_matrix < self.min_prob, 0.0, probability_matrix)

        # solve linear sum assigment for probability matrix
        row_indexes, col_indexes = linear_sum_assignment(-probability_matrix)

        # transpose probability matrix
        probability_matrix_t = probability_matrix.T

        # create output array of -1
        actual_indexes = [-1 for _ in range(probability_matrix.shape[1])]

        # create all possible numbers of indexes
        numbers = [i for i in range(len(previous_indexes))]

        # for each row index and number, if sum of row is not 0, set actual index to number
        for col_index, number in zip(col_indexes, numbers):
            if np.sum(probability_matrix_t[col_index]) != 0:
                actual_indexes[col_index] = number

        return actual_indexes

    def gather_features(self, data: pd.DataFrame) -> list[dict]:
        image_features = []
        image = self.load_image(self.images_path + data['name'])
        for bbox in data['bounding_boxes']:
            image_features.append(self.gather_bbox_features(bbox, image))
        return image_features

    def gather_bbox_features(self, bbox: list[int, int], image: np.ndarray) -> dict:
        bbox_features = {}
        bbox_image = self.cut_image(image, bbox)
        bbox_features['center'] = self.calculate_center(bbox)
        bbox_features['histogram'] = self.create_histogram(bbox_image)
        bbox_features['image'] = bbox_image
        bbox_features['bbox'] = bbox
        return bbox_features

    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        return img

    @staticmethod
    def calculate_center(bbox: list[int]) -> tuple[int, int]:
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

    @staticmethod
    def cut_image(image: np.ndarray, bbox: list[int, int, int, int]):
        x, y, w, h = bbox
        cut_image = image[y:y + h, x:x + w]
        return cut_image

    @staticmethod
    def create_histogram(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return hist

    @staticmethod
    def histogram_similarity(hist1, hist2):
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        intersection = np.minimum(hist1, hist2)
        similarity = np.sum(intersection)
        return similarity

    @staticmethod
    def compare_bounding_boxes(box1, box2):
        ratio1 = box1[2] / box1[3]
        ratio2 = box2[2] / box2[3]
        difference = abs(ratio1 - ratio2)
        similarity = 1 - difference
        return similarity

    @staticmethod
    def convert_list(nested_list):
        return "\n".join([" ".join(map(str, sublist)) for sublist in nested_list])

    def draw_image(self, img: np.ndarray, bboxes: list[list[int]]):
        img_with_bboxes = self.draw_bboxes(img, bboxes)
        self.show_img(img_with_bboxes)

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


def save_to_file(data: str):
    if len(sys.argv) > 2:
        output_file_path = sys.argv[2]
        with open(output_file_path, 'w') as file:
            file.write(data)


if __name__ == '__main__':
    tracker = PersonTracker(get_dataset_path_as_arg())
    output = tracker.run()
    save_to_file(output)
