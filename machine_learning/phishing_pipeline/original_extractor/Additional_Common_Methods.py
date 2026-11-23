def read_file_to_array(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read lines and strip newline characters
            content_array = [line.strip() for line in file.readlines()]
        return content_array
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []