import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Podaj ścieżkę do pliku jako argument!")
        sys.exit(1)

    file_path = sys.argv[1]

    print("Ścieżka do pliku:", file_path)