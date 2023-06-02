import sys
import math
import numpy as np
import pandas as pd
import cv2
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class PersonTracker:
    def __init__(self, files_path: str, bboxes_path: str = None, images_path: str = None):
        self.files_path = files_path or get_dataset_path_as_arg()
        self.bboxes_path = bboxes_path or f"{files_path}/bboxes.txt"
        self.images_path = images_path or f"{files_path}/frames/"
        self.image_shape = None

    def run(self) -> dict | None:
        data = self.load_data()
        output = self.frames_following(data)
        return output

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
        print(frames_numbers)

    def get_image_frames(self, row, before_frame_data, before_frame_numbers):
        image_frames_numbers = []
        actual_frame_data = []
        image = self.load_image(self.images_path + row['name'])
        for bbox in row['bounding_boxes']:
            actual_object_data = {}
            bbox_image = self.cut_image(image, bbox)
            actual_object_data['center'] = self.calculate_center(bbox)
            actual_object_data['histogram'] = self.create_histogram(bbox_image)
            best_probability = 0
            before_number = -1
            print(f'bbox: {bbox}')
            if before_frame_data is not None:
                for before_object_number, before_object_data in zip(before_frame_numbers, before_frame_data):
                    probability = self.calculate_probabilities(actual_object_data,
                                                               before_object_data,
                                                               image.shape)
                    print(probability)
                    if probability > 0.7 and probability > best_probability:
                        best_probability = probability
                        before_number = before_object_number + 1

            # if best_probability > 0:
            #     print(f'image: {row["name"]}, bbox: {bbox}, probability: {best_probability}')
            actual_frame_data.append(actual_object_data)
            image_frames_numbers.append(before_number)
        return image_frames_numbers, actual_frame_data

    def calculate_probabilities(self, actual_object_data, before_object_data, image_size):
        distance_probability = self.compute_distance_probability(
            actual_object_data['center'],
            before_object_data['center'],
            image_size)
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

    def compute_distance_probability(self, actual_center, before_center, img_shape) -> float:
        max_distance = self.calculate_distance((0, 0), (img_shape[1], img_shape[0]))
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

        distance_probability = 0.6 * distance_probability
        histogram_probability = 1.0 * histogram_probability

        model = BayesianNetwork([('Distance', 'Similarity'), ('Histogram', 'Similarity')])

        # Define the conditional probability distributions
        cpd_distance = TabularCPD(variable='Distance', variable_card=2,
                                  values=[[1 - distance_probability],
                                          [distance_probability]])
        cpd_histogram = TabularCPD(variable='Histogram', variable_card=2,
                                   values=[[1 - histogram_probability],
                                           [histogram_probability]])
        cpd_similarity = TabularCPD(variable='Similarity', variable_card=2, values=[[1, 0.05, 1.0, 0.2],
                                                                                    [0, 0.95, 0.0, 0.8]],
                                    evidence=['Distance', 'Histogram'], evidence_card=[2, 2])

        model.add_cpds(cpd_distance, cpd_histogram, cpd_similarity)

        inference = VariableElimination(model)
        query = inference.query(variables=['Similarity'])
        probability_same_image = query.values[1]
        return probability_same_image

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
