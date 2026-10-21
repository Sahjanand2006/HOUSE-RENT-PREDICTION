import pandas as pd

df = pd.read_csv("House_Rent_Dataset.csv")

print(df.head())

print(df.shape)
print(df.columns)
print(df.info())

print(df["Rent"].describe())
print(df["City"].value_counts())

import matplotlib.pyplot as plt

plt.hist(df["Rent"], bins=50)
plt.xlabel("Rent")
plt.ylabel("Number of House")
plt.title("Rent Distribution")
plt.show()

print(df["Rent"].quantile(0.99))
print(df["Rent"].max())

rent_limit = df["Rent"].quantile(0.99)

df_clean = df[df["Rent"] <= rent_limit].copy()

print("Original data:", df.shape)
print("Clean data:", df_clean.shape)

print("Dublicate rows:", df_clean.duplicated().sum())

print("\nArea Type:")
print(df_clean["Area Type"].value_counts())

print("\nFurnishing Status:")
print(df_clean["Furnishing Status"].value_counts())

print("\nTenant Preferred:")
print(df_clean["Tenant Preferred"].value_counts())

print('\nFloor:')
print(df_clean["Floor"].value_counts().head(15))

print("\nMissing values:")
print(df_clean.isnull().sum())

# Features and Target
X = df_clean.drop("Rent", axis=1)
y = df_clean["Rent"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('\nTraining data:', X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns
preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols), ("num", "passthrough", numeric_cols)])

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("\nEncoded training data:", X_train_encoded.shape)
print("Encoded testing data:", X_train_encoded.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)

model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("r2 Score:", r2)

#New House Rent Prediction

new_house = pd.DataFrame({
    "Posted On": ["2026-10-05"],
    "BHK": [2],
    "Size": [1000],
    "Floor": ["2 out of 4"],
    "Area Type": ["Super Area"],
    "Area Locality": ["Kormangala"],
    "City": ["Bangalore"],
    "Furnishing Status": ["Semi-Furnished"],
    "Tenant Preferred": ["Bachelors/Family"],
    "Bathroom": [2],
    "Point of Contact": ["Contact Owner"]
})

new_house_encoded = preprocessor.transform(new_house)

predicted_rent = model.predict(new_house_encoded)

print("\nNew House Details:")
print(new_house)

print("\nPredicted Rent:", predicted_rent[0])

#New House Prediction

new_house = pd.DataFrame({
    "Posted On": ["2026-10-05"],
    "BHK": [2],
    "Size": [1000],
    "Floor": ["2 out of 4"],
    "Area Type": ["Supr Area"],
    "Area Locality": ["Kormangala"],
    "City": ["Banglore"],
    "Furnishing Status": ["Semi-Furnished"],
    "Tenant Preferred": ["Bachelors/Family"],
    "Bathroom": [2],
    "Point of Contact": ["Contact Owner"]
})

new_house_encoded = preprocessor.transform(new_house)

predicted_rent = model.predict(new_house_encoded)

print("\nPredicted Rent: Rs", round(predicted_rent[0], 2))

#User Input Prediction

bhk = int(input("Enter BHK: "))
size = int(input("Enter Size(sqft) :"))
floor = input("Enter Floor (e.g. 2 out of 4): ")
area_type = input("Enter Area Locality: ")
area_locality = input("Enter Area Locality: ")
city = input("Enter City: ")
furnishing = input("Enter Furnishing Status: ")
tenant = input("Enter Tenant Prefered: ")
bathroom = int(input("Enter Bathroom: "))
contact = input("Enter point of Contact: ")

new_house = pd.DataFrame({
    "Posted On": ["2026-10-05"],
    "BHK": [bhk],
    "Size": [size],
    "Floor": [floor],
    "Area Type": [area_type],
    "Area Locality": [area_locality],
    "City": [city],
    "Furnishing Status": [furnishing],
    "Tenant Preferred": [tenant],
    "Bathroom": [bathroom],
    "Point of Contact": [contact]
})

new_house_encoded = preprocessor.transform(new_house)

predicted_rent = model.predict(new_house_encoded)

print("\nPredicted Rent: Rs", round(predicted_rent[0], 2))