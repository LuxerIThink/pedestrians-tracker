import sys
from processing.trackers import PedestriansTracker


def get_args() -> str:
    if len(sys.argv) < 2:
        print("Add files_path to dataset as first argument!")
        sys.exit(1)
    return sys.argv[1]


def indexes_to_str(nested_list: list[list[int]]):
    return "\n".join([" ".join(map(str, sublist)) for sublist in nested_list])


def save_to_file(data: str, output_path: str):
    with open(output_path, 'w') as file:
        file.write(data)


if __name__ == '__main__':
    tracker = PedestriansTracker()
    file_path = get_args()
    indexes = tracker.tracking(file_path)
    print(indexes_to_str(indexes))
    if len(sys.argv) > 2:
        save_to_file(indexes_to_str(indexes), sys.argv[2])
