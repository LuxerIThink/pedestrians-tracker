import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class PedestriansTracker:
    def __init__(self, min_probability: float = None):
        # Paths to files, set by tracking function
        self.files_path = None
        self.bboxes_path = None
        self.images_path = None

        # Parameters
        self.min_probability = min_probability or 0.275

    def tracking(self, data_folder_path: str) -> list[list[int]]:
        """
        Main function of class
        :param data_folder_path: str: path to folder with files
        :return: list[list[int]]: indexes of frames with pedestrians
        """
        self.__set_required_paths(data_folder_path)
        raw_data = self.load_data()
        indexes = self.match_data_indexes(raw_data)
        return indexes

    def __set_required_paths(self, main_path: str):
        self.files_path = main_path
        self.bboxes_path = f"{main_path}/bboxes.txt"
        self.images_path = f"{main_path}/frames/"

    def load_data(self) -> pd.DataFrame:
        with open(self.bboxes_path, 'r') as file:
            lines = file.readlines()

        # Default values
        data = []
        bboxes_count = 0
        row = []
        bboxes = []

        for line in lines:

            # Strip lines from whitespaces
            line = line.strip()

            # Load img name and add image_name to row
            if bboxes_count == 0:
                row = [line]
                bboxes = []
                bboxes_count = -1

            # Load number of bboxes
            elif bboxes_count == -1:
                bboxes_count = int(line)

            # Load bboxes and convert bbox string to list of floats
            else:
                bbox_points = [int(float(number)) for number in line.split()]
                bboxes.append(bbox_points)
                bboxes_count -= 1

                # If all bboxes loaded, append row to data
                if bboxes_count == 0:
                    row.append(bboxes)
                    data.append(row)

        df = pd.DataFrame(data, columns=['name', 'bboxes'])

        return df

    def match_data_indexes(self, data: pd.DataFrame) -> list[list[int]]:
        # Set empty values
        previous_data = None
        previous_indexes = []
        indexes = []

        # Track indexes for frames, between current and previous frame
        for _, frame_data in data.iterrows():
            actual_data = self.create_features_per_image(frame_data)
            previous_indexes, previous_data = self.match_image_indexes(actual_data, previous_data, previous_indexes)
            indexes.append(list(previous_indexes))

        return indexes

    def create_features_per_image(self, data: pd.Series) -> list[dict]:
        img = cv2.imread(self.images_path + data['name'])
        img_features = []
        for bbox in data['bboxes']:
            img_features.append(self.create_features_per_clipping(img, bbox))
        return img_features

    def create_features_per_clipping(self, img: np.ndarray, bbox: list[int]) -> dict:
        clipping_img = self.clip_img(img, bbox)
        clipping_features = {
            'bbox': self.convert_bbox_to_points(bbox),
            'histogram': self.create_histogram(clipping_img),
            'image': clipping_img
        }
        return clipping_features

    @staticmethod
    def clip_img(img: np.ndarray, bbox: list[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        clipping = img[y:y + h, x:x + w]
        return clipping

    @staticmethod
    def create_histogram(img: np.ndarray) -> np.ndarray:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        return histogram

    def match_image_indexes(self, current_data: list[dict], previous_data: list[dict],
                            previous_indexes: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        # Check is the first frame
        if previous_data is not None:
            probability_matrix = self.create_probability_matrix(current_data, previous_data)
            current_indexes = self.fit_objects(previous_indexes, probability_matrix)
        else:
            current_indexes = np.full(len(current_data), -1)

        return current_indexes, current_data

    def create_probability_matrix(self, current_data: list[dict], previous_data: list[dict]) -> np.ndarray:
        # Create empty probability matrix
        probability_matrix = np.zeros((len(previous_data), len(current_data)))
        # Calculate probability between actual and previous clipping
        for i, previous_clipping in enumerate(previous_data):
            for j, actual_clipping in enumerate(current_data):
                probability_matrix[i, j] = self.clippings_similarity(actual_clipping, previous_clipping)
        # Find indexes with the highest probability
        return probability_matrix

    def clippings_similarity(self, current_clipping: dict, previous_clipping: dict) -> float:
        images_similarity = self.images_similarity(previous_clipping['image'], current_clipping['image'])
        histograms_similarity = self.compare_histograms(previous_clipping['histogram'], current_clipping['histogram'])
        iou_similarity = self.iou_similarity(previous_clipping['bbox'], current_clipping['bbox'])
        similarity = (images_similarity + histograms_similarity + iou_similarity)/3
        print(similarity)
        return similarity

    @staticmethod
    def images_similarity(img1: np.ndarray, img2: np.ndarray,
                          size: tuple[int, int] = (128, 128)) -> float:
        # Resize the images to a consistent size for comparison
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)

        # Apply template Matching
        res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, similarity, _, _ = cv2.minMaxLoc(res)

        return similarity

    @staticmethod
    def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        intersection = np.minimum(hist1, hist2)
        similarity = float(np.sum(intersection))
        return similarity

    def iou_similarity(self, bbox1: list[tuple[int, int]], bbox2: list[tuple[int, int]]) -> float:
        intersection = self.calculate_intersection(bbox1, bbox2)
        union = self.calculate_union(intersection, bbox1, bbox2)
        if union != 0:
            similarity = intersection / union
            if similarity < 0:
                similarity = 0
        else:
            similarity = 0
        return similarity

    @staticmethod
    def convert_bbox_to_points(bbox: list[int]) -> list[tuple[int, int]]:
        x_min, y_min, x_max, y_max = bbox
        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        return points

    @staticmethod
    def calculate_intersection(bbox1: list[tuple[int, int]], bbox2: list[tuple[int, int]]) -> float:
        x_left = max(bbox1[0][0], bbox2[0][0])
        y_top = max(bbox1[0][1], bbox2[0][1])
        x_right = min(bbox1[2][0], bbox2[2][0])
        y_bottom = min(bbox1[2][1], bbox2[2][1])

        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        return intersection_area

    @staticmethod
    def calculate_union(intersection: float, bbox1: list[tuple[int, int]], bbox2: list[tuple[int, int]]) -> float:
        area_bbox1 = (bbox1[2][0] - bbox1[0][0]) * (bbox1[2][1] - bbox1[0][1])
        area_bbox2 = (bbox2[2][0] - bbox2[0][0]) * (bbox2[2][1] - bbox2[0][1])
        union_area = area_bbox1 + area_bbox2 - intersection
        return union_area

    def __fit_objects(self, previous_indexes: np.ndarray, probability_matrix: np.ndarray[float]):
        """
        Fit objects from previous frame to current frame
        :param previous_indexes: m previous indexes
        :param probability_matrix: (n x m) [0-1] probabilities [n - current clipping, m - previous clipping]
        :return: current indexes [-1 - n], [-1 - new index, n - nth previous index]
        """
        # Zeroing values if they are below the minimum probability
        probability_matrix = np.where(probability_matrix < self.min_probability, 0.0, probability_matrix)

        # Solve linear sum assigment for probability matrix
        row_indexes, col_indexes = linear_sum_assignment(-probability_matrix)

        # Transpose probability matrix
        probability_matrix_t = probability_matrix.T

        # Create base of actual images made of -1 for each row
        current_indexes = [-1 for _ in range(probability_matrix.shape[1])]

        # Create all possible possible_indexes of indexes based on previous indexes
        possible_indexes = [i for i in range(len(previous_indexes))]

        # For each row index and number, if sum of row is not 0, set actual index to number
        for col_index, number in zip(col_indexes, possible_indexes):
            if np.sum(probability_matrix_t[col_index]) != 0:
                current_indexes[col_index] = number

        return current_indexes
