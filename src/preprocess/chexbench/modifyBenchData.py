import pandas


# TODO - Add more preprocessing steps as needed, the chex becnh findings in FInsdings generation tasks needs to be changed with the originals in mimic cxr
if __name__ == "__main__":
    # Load the data

    df = pandas.read_csv("data/chexbench/chexbench.csv")

    # Remove the 'id' column
    df = df.drop(columns=["id"])

    # Save the modified DataFrame to a new CSV file
    df.to_csv("data/chexbench/chexbench_modified.csv", index=False)
    print("Data preprocessing complete. Modified data saved to 'data/chexbench/chexbench_modified.csv'.")
    # Note: The above code assumes that the 'id' column is present in the original CSV file.
    # If the 'id' column is not present, you can skip the drop step.
    # If you need to perform additional preprocessing steps, you can add them here.