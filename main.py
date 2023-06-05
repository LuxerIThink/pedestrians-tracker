import sys
import math
import numpy as np
import pandas as pd
import cv2
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.optimize import linear_sum_assignment


class PersonTracker:
    def __init__(self, files_path: str, bboxes_path: str = None, images_path: str = None):
        self.files_path = files_path or get_dataset_path_as_arg()
        self.bboxes_path = bboxes_path or f"{files_path}/bboxes.txt"
        self.images_path = images_path or f"{files_path}/frames/"
        self.image_shape = None
        self.min_prob = 0.8

    def run(self) -> dict | None:
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
        before_frame_data = None
        before_frame_numbers = []
        for _, row in df.iterrows():
            output = self.get_image_frames(row, before_frame_data, before_frame_numbers)
            frame_numbers, before_frame_data = output
            before_frame_numbers = frame_numbers
            frames_numbers.append(frame_numbers)
        return frames_numbers

    def get_image_frames(self, image_data, before_data, before_numbers):
        actual_numbers = []
        actual_data = self.gather_features(image_data)
        probability_matrix = []
        if before_data is not None:
            for before_object in before_data:
                probability_for_before_object = []
                for actual_object in actual_data:
                    probability = self.calc_probab(actual_object, before_object)
                    probability_for_before_object.append(probability)
                probability_matrix.append(probability_for_before_object)
            probability_matrix = np.array(probability_matrix)
            actual_numbers = self.fit_objects(before_numbers, probability_matrix)
        else:
            actual_numbers = np.array([-1 for _ in range(len(actual_data))])
        print(actual_numbers)
        return actual_numbers, actual_data

    def fit_objects(self, before_numbers, prob_matrix):
        prob_matrix = np.where(prob_matrix > self.min_prob, prob_matrix, 0.0)
        num_rows, num_cols = prob_matrix.shape
        if num_cols > num_rows:
            prob_matrix = np.pad(prob_matrix, ((0, num_cols - num_rows), (0, 0)), mode='constant')
        before_numbers += 1
        num_rows_diff = prob_matrix.shape[0] - before_numbers.shape[0]
        before_numbers = np.pad(before_numbers, (0, num_rows_diff), mode='constant', constant_values=-1)
        _, col_ind = linear_sum_assignment(-prob_matrix)
        before_numbers[prob_matrix[:, -1] == 0.0] = -1
        actual_numbers = before_numbers[col_ind]
        return actual_numbers

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
        return bbox_features

    def calc_probab(self, actual_object_data, before_object_data):
        distance_probability = self.compute_distance_probability(
            actual_object_data['center'],
            before_object_data['center'])
        compare_histograms = self.compute_histogram_similarity(
            actual_object_data['histogram'],
            before_object_data['histogram'])
        # print(f'distance_probability: {distance_probability}\n'
        #       f'compare_histograms: {compare_histograms}')
        probability = self.calculate_similarity(distance_probability, compare_histograms)
        return probability

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

    def compute_distance_probability(self, actual_center, before_center) -> float:
        max_distance = self.calculate_distance((0, 0), (self.image_shape[1], self.image_shape[0]))
        distance = self.calculate_distance(actual_center, before_center)
        normalized_distance = distance / max_distance
        probability = 1 - normalized_distance
        return probability

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
    def compute_histogram_similarity(hist1, hist2):
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        intersection = np.minimum(hist1, hist2)
        similarity = np.sum(intersection)
        return similarity

    def calculate_similarity(self, distance_probability, histogram_probability):

        distance_probability = 0.7 * distance_probability
        histogram_probability = 1.0 * histogram_probability

        model = BayesianNetwork([('Distance', 'Similarity'), ('Histogram', 'Similarity')])

        # Define the conditional probability distributions
        cpd_distance = TabularCPD(variable='Distance', variable_card=2,
                                  values=[[1 - distance_probability],
                                          [distance_probability]])
        cpd_histogram = TabularCPD(variable='Histogram', variable_card=2,
                                   values=[[1 - histogram_probability],
                                           [histogram_probability]])
        cpd_similarity = TabularCPD(variable='Similarity', variable_card=2, values=[[1, 0.3, 0.6, 0.15],
                                                                                    [0, 0.7, 0.4, 0.85]],
                                    evidence=['Distance', 'Histogram'], evidence_card=[2, 2])

        model.add_cpds(cpd_distance, cpd_histogram, cpd_similarity)

        inference = VariableElimination(model)
        query = inference.query(variables=['Similarity'])
        probability_same_image = query.values[1]
        return probability_same_image

    def convert_list(self, nested_list):
        return "\n".join([" ".join(map(str, sublist)) for sublist in nested_list])

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


def save_to_file(data: dict):
    if len(sys.argv) > 2:
        output_file_path = sys.argv[2]
        with open(output_file_path, 'w') as file:
            file.write(data)


if __name__ == '__main__':
    tracker = PersonTracker(get_dataset_path_as_arg())
    output = tracker.run()
    save_to_file(output)