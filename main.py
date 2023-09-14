import csv
import readingFileUtils
file_path = "AirplaneCrashes.csv"

# main

i = 2
j = 3
try:
    readingFileUtils.getVariable(i, j, file_path)
except FileNotFoundError:
    print("File not found or cannot be created.")
except PermissionError:
    print("Permission denied to create or write to the file.")
except Exception as e:
    print(f"An error occurred: {e}")
