import pandas as pd

myCSV = pd.read_csv("Data/wwIIAircraft.csv")


# Comment out some prints to make it easier to see hehe
# print(myCSV)

# Lots of info = Let's only print out the first 10 entries
# print(myCSV.head(10))
# print(myCSV.tail(10))

# print(myCSV.dtypes)

# List all the aircraft that were made by Germany
german_aircraft = myCSV[myCSV["Country of Origin"] == "Germany"]
print(german_aircraft)
# using the to_string method will print them all instead of showing ..... like the above
# print(german_aircraft.to_string())

pdGermany = myCSV.loc[(myCSV["Country of Origin"] == "Germany") & (myCSV["Year in Service"] == 1942)]
print(pdGermany.to_string())

# Anna is an absolutely stunning woman.