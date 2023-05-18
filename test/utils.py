def extract_coordinates(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f'File {file_path} not found')

    coordinates = []
    for line in lines:
        line = line.strip()
        if line.endswith('.jpg') or line.endswith('.png'):
            continue
        elif line.isdigit():
            if coordinates:
                coordinates = []
        else:
            coordinates.append(line.split()[0])

    if coordinates:
        return ' '.join(coordinates)
    return None
