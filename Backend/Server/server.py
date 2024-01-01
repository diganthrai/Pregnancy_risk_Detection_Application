from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os

app = Flask(__name__)

# cred = credentials.Certificate('path/to/serviceAccountKey.json')
# firebase_admin.initialize_app(cred)
# db = firestore.client()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Backend/Server/mai-care-firebase-adminsdk-sfqxu-962ad593ad.json"
db=firestore.Client()


@app.route('/register', methods=['POST'])
def register():
    # extract user data from request body
    data = request.json
    name = data['name']
    trimester = data['trimester']
    age = data['age']
    email = data['email']
    height = data['height']
    weight = data['weight']
    
    doc_ref = db.collection('users').document(email)
    doc_ref.set({
        'name': name,
        'trimester': trimester,
        'age': age,
        'height': height,
        'weight': weight
    })
    
    return jsonify({'message': 'User registration successful!'})
    
if __name__ == '__main__':
    app.run()
