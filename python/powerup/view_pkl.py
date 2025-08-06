import pickle
import pprint

def explore_tetris_data(file_path):
    """
    Loads a .pkl file and provides a more detailed exploration
    if the top-level data structure is a dictionary.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Successfully loaded data from: {file_path}")
            print("\n--- Data Type ---")
            print(type(data))

            if isinstance(data, dict):
                print(f"It's a dictionary with {len(data)} keys.")
                print("\n--- Dictionary Keys ---")
                for key in data.keys():
                    print(f"- {key}")

                print("\n--- Exploring Key Values ---")
                for key, value in data.items():
                    print(f"\n--- Data for Key: '{key}' ---")
                    print(f"Type of value for '{key}': {type(value)}")

                    # For lists, show the length and the type of the first item
                    if isinstance(value, list):
                        print(f"It's a list with {len(value)} items.")
                        if len(value) > 0:
                            print(f"Type of first item: {type(value[0])}")
                            print("Sample of first item:")
                            pprint.pprint(value[0]) # Print the first element for inspection
                        else:
                            print("The list is empty.")
                    # For dictionaries, show length and sample the first few key-value pairs
                    elif isinstance(value, dict):
                        print(f"It's a dictionary with {len(value)} keys.")
                        print("Sample of first 3 key-value pairs:")
                        sample_count = 0
                        for sub_key, sub_value in value.items():
                            if sample_count < 3:
                                print(f"  - Sub-key: '{sub_key}', Type: {type(sub_value)}")
                                # You might print sub_value if it's simple, or just its type
                                # pprint.pprint(sub_value)
                                sample_count += 1
                            else:
                                break
                    # For NumPy arrays, show shape and a small sample
                    elif 'numpy' in str(type(value)): # Checks if it's a numpy array without importing numpy
                        print(f"It's likely a NumPy array with shape: {value.shape}")
                        if value.size > 0:
                            print("Sample of array content (first few elements or rows):")
                            # This part is tricky without knowing the exact shape,
                            # but for 2D arrays, showing the first row is common.
                            if value.ndim == 1 and value.size > 5:
                                print(value[:5])
                            elif value.ndim == 2 and value.shape[0] > 1:
                                print(value[0]) # First row
                            else:
                                print(value) # Print whole array if small
                    # For other types, try to pretty print directly
                    else:
                        pprint.pprint(value)

            else:
                print("The top-level data is not a dictionary. Here's a direct print:")
                pprint.pprint(data)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pickle.UnpicklingError as e:
        print(f"Error unpickling the file: {e}")
        print("This might happen if the file is corrupted or not a valid pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- How to use it ---
# Make sure 'large_realistic_tetris_dataset.pkl' is in the same directory
# as your Python script, or provide the full path to the file.
file_path = 'tetris_boards3.pkl'
explore_tetris_data(file_path)
