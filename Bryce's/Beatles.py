import pandas as pd

pdB = pd.DataFrame(
    {
        "Names": pd.Series(["Paul", "Ringo", "George", "John"]),
        "Status": pd.Categorical(["Alive", "Alive", "Dead", "Dead"]),
        "Plays": pd.Series (["Sings", "Drums", "Base", "Sings"]),
        "Band": "Beatles"
    }
)
# print(pdB)


# Print out the first 2 Rows of this
# print(pdB.loc[:1])
# Just print out the third dude
# print (pdB.loc[2:2])
# or
# print(pdB.loc[2])


# Get all of the singers in the band - if the category is sings
pdSings = pdB[pdB["Plays"] == "Sings"]
# print("\n\nSingers")
# print(pdSings)

print("\n\nPlays Column")
# Get the values in the Plays column
print(pdB["Plays"])