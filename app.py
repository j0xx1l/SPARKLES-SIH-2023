# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import instaloader
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Function to get Instagram data using Instaloader
def get_instagram_data(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        return {
            "userFollowerCount": profile.followers,
            "userFollowingCount": profile.followees,
            "userBiographyLength": len(profile.biography),
            "userMediaCount": profile.mediacount,
            "userHasProfilPic": int(not profile.is_private and profile.profile_pic_url is not None),
            "userIsPrivate": int(profile.is_private),
            "usernameDigitCount": sum(c.isdigit() for c in profile.username),
            "usernameLength": len(profile.username),
        }
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile with username '{username}' not found.")
        return None

# Load the trained model
load_model = tf.keras.models.load_model('trainedmodel')

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html')

@app.route('/result')
def result():
    result_text = request.args.get('result')
    return render_template('result.html', result_text=result_text)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the username from the form submission
        username = request.form['username']

        # Get Instagram data
        insta_data = get_instagram_data(username)

        if insta_data:
            # Convert Instagram data to NumPy array
            X_new = np.array([list(insta_data.values())], dtype=np.float32)

            # Make predictions
            predictions = load_model.predict(X_new)

            # Determine the result
            result_text = f"Prediction for {username}: {'Fake' if predictions[0][0] >= 0.5 else 'Real'} (Probability: {predictions[0][0]:.4f})"

            # Redirect to the result page with the result as a parameter
            return redirect(url_for('result', result=result_text))

        else:
            return jsonify({"result": f"Profile with username '{username}' not found."})
    except Exception as e:
        return jsonify({"result": f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5003)
