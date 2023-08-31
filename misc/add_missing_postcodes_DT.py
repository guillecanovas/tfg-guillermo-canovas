import csv
import pandas as pd
from pandas.core.frame import DataFrame

if __name__ == '__main__':
    # Read CSV as a list of lists (all elements of the inner lists are Strings)
    with open('./script_results/solar_hours_by_postcode.csv', newline='') as f:
        reader = csv.reader(f)
        lines = list(reader)

    # Remove first line (header)
    lines.remove(["postcode", "DT", "is_sunny?"])

    # Number of lines of original CSV
    number_of_line = 0
    MAX_LINES = len(lines)

    # Creating a dict (more efficient than appending to a dataframe in each iteration)
    d = dict()

    # Number of lines of the resulting CSV
    resulting_csv_lines = 0

    # Initializing variables
    last_postcode = 0
    last_DT = 0
    last_boolean = 0

    # Iterating over original CSV 
    for line in lines:
        # Can't keep iterating
        if MAX_LINES == number_of_line + 1:
            break

        # Taking current line values
        current_postcode = line[0]
        current_DT = int(line[1]) % 100
        current_prefix_DT = int(int(line[1]) / 100)
        current_boolean = line[2]

        # Taking next line values
        next_line = lines[number_of_line + 1]
        next_postcode = next_line[0]
        next_DT = int(next_line[1]) % 100
        next_boolean = next_line[2]

        # Add the current line to the dict
        d[resulting_csv_lines] = {
                                'postcode': current_postcode,
                                'DT': int(current_prefix_DT * 100 + current_DT),
                                'is_sunny?': current_boolean 
                                }
        resulting_csv_lines += 1

        # There is an empty space...
        if current_DT + 1 != next_DT and current_DT != 48 and last_postcode == current_postcode and current_postcode == next_postcode:
            if last_boolean == current_boolean and current_boolean == next_boolean:
                    # Add a line to the empty space
                    d[resulting_csv_lines] = {
                                            'postcode': current_postcode,
                                            'DT': int(current_prefix_DT * 100 + current_DT + 1),
                                            'is_sunny?': current_boolean 
                                            }
                    resulting_csv_lines += 1
            else:
                print('Check manually line: ' + str(current_postcode) + " " + str(int(current_prefix_DT * 100 + current_DT + 1)))
                for i in range(23240, 23248 + 1):
                    d[resulting_csv_lines] = {
                                            'postcode': current_postcode,
                                            'DT': i,
                                            'is_sunny?': current_boolean 
                                            }
                    resulting_csv_lines += 1

        # Updating loop variables
        last_postcode = current_postcode
        last_DT = current_DT
        last_boolean = current_boolean
        number_of_line += 1

    # Generating and exporting dataframe from dict
    df = DataFrame.from_dict(d, "index")
    df.to_csv('./script_results/solar_hours_by_postcode_fixed.csv', index=False, header=True)
