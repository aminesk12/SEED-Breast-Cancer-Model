import pandas as pd

# Load the dataset
df = pd.read_csv(r'./SEER Breast Cancer Dataset .csv', encoding='ascii')
df.head()

# Define the function to get data where patients are 'Dead'
def get_Dead_Data(data):
    dead_data = data[data['Status'] == 'Dead']  # Replace 'Survival Status' with the actual column name
    return dead_data

# Use the function to get the data
dead_data = get_Dead_Data(df)

# Export the filtered data to a CSV file
dead_data.to_csv('dead_data.csv', index=False)

print("Filtered data has been exported to 'dead_data.csv'.")
