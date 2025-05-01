import os
import pandas as pd
import argparse

def generate_cumulated_average_summary(input_dir, output_file):
    """
    Reads multiple profile summary CSV files from the input directory,
    calculates the cumulative average for each reduction type, and writes
    the result to the output file.

    Args:
        input_dir (str): Directory containing the profile summary CSV files.
        output_file (str): Path to the output CSV file.
    """
    # Initialize an empty DataFrame to store cumulative data
    cumulative_data = pd.DataFrame()

    # Iterate through all CSV files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing file: {file_path}")
            # Read the current CSV file
            data = pd.read_csv(file_path)
            # Append the data to the cumulative DataFrame
            cumulative_data = pd.concat([cumulative_data, data], ignore_index=True)

    # Group by 'Reduction Type' and calculate the average for numeric columns
    average_summary = cumulative_data.groupby("Reduction Type", as_index=False).mean()

    # Write the cumulative average summary to the output file
    average_summary.to_csv(output_file, index=False)
    print(f"Cumulative average summary written to {output_file}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate cumulative average summary from profile CSV files.")
    parser.add_argument("-d", "--directory", required=True, help="Directory containing the profile summary CSV files.")
    args = parser.parse_args()

    # Input directory containing profile summary CSV files
    input_directory = args.directory
    # Output file for the cumulative average summary
    output_csv = os.path.join(input_directory, "cumulated_average_summary.csv")

    # Generate the cumulative average summary
    generate_cumulated_average_summary(input_directory, output_csv)