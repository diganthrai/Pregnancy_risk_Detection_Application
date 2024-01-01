import pickle
with open('Backend/ML/Random_Forest/model.pkl', 'rb') as file:
    model = pickle.load(file)


#more sensitive to systolic and blood sugar
user_input = [[25, 130, 80, 126, 98.6, 80]]
risk_level = model.predict(user_input)
print("Predicted risk level:", risk_level[0])
if risk_level==0:
    print("Low Risk")
else:
    print("High Risk")
