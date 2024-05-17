import pandas as pd

pdBB = pd.read_csv("Data/mlb_salaries.csv")

# Create a DataFrame of only the players who played for Toronto
pdTOR = pdBB[pdBB["teamid"] == "TOR"]

# Print data types and head of the original DataFrame
print(pdBB.dtypes)
print(pdBB.head().to_string())

# Aggregate count of rows in the Toronto DataFrame
print(pdTOR.aggregate("count"))

# Calculate and print average salary for Toronto
print("Average salary for Toronto")
print(pdTOR["salary"].aggregate("average"))

# Find the maximum salary and corresponding player name
pdMax = pdTOR[["salary"]].max()
maxSal = float(pdMax["salary"])
pdPlayer = pdTOR[pdTOR["salary"] == maxSal]["player_name"]

# Display the player with the maximum salary
print(f"\nThe player {pdPlayer.values[0]} with the max salary is ${maxSal:.2f}")

# Count the number of players on the Blue Jays that bat right
pdRight = pdTOR[pdTOR["bats"] == "R"].aggregate("count")

print(f"The amount of right batters is {pdRight['bats']}")
