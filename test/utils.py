
def load_text_file(file_path: str) -> list:
    try:
        with open(file_path, 'r'):
            output = [[int(num) for num in line.split()] for line in open(file_path, 'r').read().splitlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f'File {file_path} not found')
    return output


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