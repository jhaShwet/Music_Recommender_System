# Music_Recommender_System

A basic full stack project using Collaborative filtering approach and Deep Learning.

# Project Components

● HTML Files: For creating the user interface and presenting content to users.
● JavaScript File (script.js): Handles client-side scripting and provides interactivity to the user interface.
● Python Files (app.py, recommend.py): Manage the server-side scripting and backend logic, including data processing and recommendation algorithm implementation.
● Flask (for backend): Implementation of a basic web application using the Flask framework in Python to recommend songs to users based on their preferences. It defines two application routes. The first route is responsible for accessing the HTML form at the root URL and it serves as the interface for users, while the other one handles the recommendation logic and processes the input and generates personalized song recommendations.
Reason: Flask is a lightweight web application in Python. It provides tools, libraries, and technologies for creating web applications efficiently and the variety of extensions and libraries.

Note: Setting the host to '0.0.0.0' and the port to 5000, making the application accessible on the local network.

# Working

The collaborative filtering-based recommendation system uses Keras, which makes use of the machine learning technique: matrix factorization (Matrix factorization decomposes the original user-item interaction matrix into two lower-dimensional matrices i.e. user matrix and an item matrix. The dimensions of these matrices are then chosen such that the product of the two matrices closely approximates the original matrix.) through an embedding layer. The model is trained on the dataset and then used to generate recommendations for a given user.  It first preprocesses the data, including creating mappings for user and song indices, splitting the data into training and testing sets, and then builds a deep learning model for recommendation (It creates input layers for user and song indices, embedding layers for user and song data, and performs dot product and flattening operations. It then compiles the model with the mean squared error (MSE) loss function and the Adam optimizer)

1. Importing Libraries: The code starts by importing necessary libraries such as pandas, numpy and various other components from the Keras library.
2. Defining the function to recommend songs: This function takes several parameters such as user_id, data_path, embedding_dim (the dimensionality of the embedding space), 
   epochs (100), batch_size (32) and N (the number of songs to recommend).
 
   The function performs the following steps:

   ● Loads and preprocesses the data which includes creating mappings for user and song indices, and splitting the data into training and testing sets.
   ● Building the Deep Learning Model: The function constructs a neural network model using Keras (reason: it provides high-level building blocks for constructing neural 
     networks, such as layers, models, and optimizers. These abstractions make it easier to define and train complex models without the need for extensive knowledge of the 
     underlying mathematical concepts
   ● Training the Model  using the user and song indices as inputs and other parameters as the target variable.
   ● Generating Recommendations: After training, the function uses the trained model to generate recommendations for the given user. It first finds the index of the user in 
     the dataset, identifies the songs that the user has not rated, predicts the ratings for these songs using the model, and selects the top 'N' no. of songs with the 
   highest predicted ratings.

3. Main Script: calls the function with the parameters considered for obtaining a list of recommended songs and finally, it prints the top recommended songs for the given user.
