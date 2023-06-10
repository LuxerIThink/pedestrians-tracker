import numpy as np
import cv2


def extract_solution(file_path: str) -> list[list[int]]:

    with open(file_path, 'r'):
        data = open(file_path, 'r').read().splitlines()

    output = []
    temp_numbers = []

    for line in data:
        line = line.strip()

        if line.endswith('.jpg') or line.endswith('.png'):
            if temp_numbers:
                output.append(temp_numbers)
                temp_numbers = []

        elif line.isdigit():
            continue

        else:
            temp_numbers.append(int(line.split()[0]))

    if temp_numbers:
        output.append(temp_numbers)

    return output


def save_solution(filename_path: str, data: list[list[int]]):
    with open(filename_path, 'w') as file:
        for sublist in data:
            line = ' '.join(str(num) for num in sublist)
            file.write(line + '\n')


def draw_bboxes(img: np.ndarray, bboxes: list[list[int]]) -> np.ndarray:
    output_img = img.copy()
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return output_img


def show_img(img: np.ndarray):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_img_with_tags(img: np.ndarray, bboxes: list[list[int]]):
    img_with_bboxes = draw_bboxes(img, bboxes)
    show_img(img_with_bboxes)
