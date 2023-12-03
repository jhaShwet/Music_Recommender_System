from flask import Flask, request, render_template
from recommend import recommend_songs_for_user

app = Flask(__name__)

# Replace 'path_to_dataset.csv' with the actual path to your dataset
data_path = 'Spotify_final_dataset.csv'

# Define a basic route to handle the root URL and render the form
@app.route('/')
def root():
    return render_template('recommend_form.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if user_id is None:
        return render_template('recommend_form.html', error="Please provide a valid user_id.")

    try:
        user_id = int(user_id)  # Ensure the user_id is an integer
    except ValueError:
        return render_template('recommend_form.html', error="Invalid user_id format.")

    recommended_songs = recommend_songs_for_user(user_id, data_path)
    return render_template('recommend_form.html', recommended_songs=recommended_songs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
