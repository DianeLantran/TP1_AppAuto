import csv
import pandas as pd

def getVariable(i, j, file_path):
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)

        # Skip to the desired row (i-1 times)
        for _ in range(i - 1):
            next(csv_reader)

        # Read the row you're interested in
        selected_row = next(csv_reader, None)

        if selected_row is not None:
            # Check if the selected row has enough columns
            if j <= len(selected_row):
                # Retrieve the j-th column (j-1 index)
                selected_cell = selected_row[j - 1]
                print(f"Value at row {i}, column {j}: {selected_cell}")
                return (selected_cell)
            else:
                print(f"Row {i} does not have {j} columns.")
        else:
            print(f"Row {i} does not exist in the CSV file.")


