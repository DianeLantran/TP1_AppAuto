import pandas as pd
import readingFileUtils
import preprocessing as prep

FILE_PATH = "data/AirplaneCrashes.csv"

df = pd.read_csv (FILE_PATH)

# Preprocessing
df = prep.simplifyDate(df)
df = prep.simplifyLocation(df)
df = prep.simplifyTime(df)
df = prep.colToOrdinal(df, ["Location", "Operator"])
# main

i = 2
j = 3
try:
    readingFileUtils.getVariable(i, j, FILE_PATH)
except FileNotFoundError:
    print("File not found or cannot be created.")
except PermissionError:
    print("Permission denied to create or write to the file.")
except Exception as e:
    print(f"An error occurred: {e}")
