
def load_text_file(file_path: str) -> list:
    try:
        with open(file_path, 'r') as file:
            output = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f'File {file_path} not found')
    return output


def extract_solution(file_path: str) -> str:
    data = load_text_file(file_path)
    temp_output_line = []
    output = []
    for line in data:
        line = line.strip()
        if line.endswith('.jpg') or line.endswith('.png'):
            continue
        elif line.isdigit():
            if temp_output_line:
                output.append(' '.join(temp_output_line))
                temp_output_line = []
        else:
            temp_output_line.append(line.split()[0])
    if temp_output_line:
        output.append(' '.join(temp_output_line))
    return '\n'.join(output) if output else None

