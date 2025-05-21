import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load the dataset from the uploaded CSV

df=pd.read_csv(r"D:\course\cource\diploma\term2\python588\\train.csv")
print(df.head())
print(df.info())

 
# Group by Region and Brand to count the number of units sold
top_brands_by_region = df.groupby(["Region", "Brand"])["Units_Sold"].sum().reset_index()

# Group by Region and Vehicle Type to count the number of units sold
top_vehicle_types_by_region = df.groupby(["Region", "Vehicle_Type"])["Units_Sold"].sum().reset_index()

# Sort and display top results
print("Top Brands by Region:")
print(top_brands_by_region.sort_values(by=["Region", "Units_Sold"], ascending=[True, False]).head(10))

# print("\nTop Vehicle Types by Region:")
print(top_vehicle_types_by_region.sort_values(by=["Region", "Units_Sold"], ascending=[True, False]).head(10))




#  Classify vehicle type using features like battery capacity, brand, and region
 
# Select relevant features for classification
features = ["Battery_Capacity_kWh", "Brand", "Region"]
target = "Vehicle_Type"

# Encode categorical variables
label_encoders = {}
for col in features + [target]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

# Train a classification model
model = RandomForestClassifier(n_estimators=100 )
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Example prediction for a new data point
new_data = pd.DataFrame([[80, "Toyota", "North America"]], columns=features)
for col in features:
    new_data[col] = label_encoders[col].transform(new_data[col])

predicted_type = model.predict(new_data)
print(f"Predicted Vehicle Type: {label_encoders[target].inverse_transform(predicted_type)[0]}")



