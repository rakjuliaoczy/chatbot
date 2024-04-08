import json
import csv

# Load JSON data from a file
with open('data/output.json', 'r') as json_file:
    data = json.load(json_file)

# Specify the CSV file path and open it for writing
csv_file_path = 'data/output.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    # Create a CSV writer object
    writer = csv.writer(csv_file)

    # Write the header row based on the keys of the first JSON object
    writer.writerow(data[0].keys())

    # Write each JSON object as a row in the CSV file
    for row in data:
        writer.writerow(row.values())
