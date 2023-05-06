
from bs4 import BeautifulSoup as SOUP
import re
import requests as HTTP
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from rec import recommend_music
from imdb import IMDb

ia = IMDb()
app = Flask(__name__)
CORS(app)

# Main Function for scraping


def main(emotion):

    if (emotion == "Sad" or emotion == "sad"):
        urlhere = 'https://www.imdb.com/search/title/?title_type=movie&genres=romance,comedy&sort=num_votes,desc&explore=title_type,genres'
    elif (emotion == "Happy" or emotion == "happy"):
        urlhere = 'https://www.imdb.com/search/title/?title_type=movie&genres=sci-fi,action&sort=num_votes,desc&explore=title_type,genres'
    elif (emotion == "Bored" or emotion == "bored"):
        urlhere = 'https://www.imdb.com/search/title/?title_type=movie&genres=mystery,thriller&sort=num_votes,desc&explore=title_type,genres'
    elif (emotion == "Angry" or emotion == "angry"):
        urlhere = 'https://www.imdb.com/search/title/?title_type=feature&genres=musical&sort=num_votes,desc'
    elif (emotion == "Stressed" or emotion == "stressed"):
        urlhere = 'https://www.imdb.com/search/title/?title_type=movie&genres=fantasy,adventure&sort=num_votes,desc&explore=title_type,genres'
    # HTTP request to get the data of
    # the whole page
    response = HTTP.get(urlhere)
    data = response.text

    # Parsing the data using
    # BeautifulSoup
    soup = SOUP(data, "lxml")

    # Extract movie details from the
    # data using regex
    movies = []
    for movie in soup.find_all("div", class_="lister-item-content"):
        title = movie.find_all("a")[0].get_text()
        year = movie.find_all("span", class_="lister-item-year")[0].get_text()
        genre = movie.find_all("span", class_="genre")[0].get_text().strip()
        rating = movie.find_all(
            "div", class_="ratings-bar")[0].find_all("strong")[0].get_text()
        director = movie.find_all("p")[2].find_all("a")[0].get_text()
        cast_list = movie.find_all("p")[2].find_all("a")
        cast = [", ".join(a.get_text() for a in cast_list[1:4])]
        imdb_link = "https://www.imdb.com" + movie.find_all("a")[0]["href"]
        '''
        movie_id = re.search(r'/title/(\w+)/', imdb_link).group(1)
        name = ia.get_movie(movie_id[2:])
        poster = name.data['cover url']
        '''
        movies.append({"title": title, "year": year, "genre": genre, "rating": rating,
                      "cast": cast, "director": director, "imdb_link": imdb_link})

    random.shuffle(movies)
    return movies


@app.route('/movies', methods=['GET'])
def get_movies_by_emotion():
    # Get the emotion parameter from the request
    emotion = request.args.get('emotion')
    print(emotion)
    # Get the array of movies associated with the emotion
    movies = main(emotion)

    # Return the array of movies in a JSON response
    return jsonify({'movies': movies}), 200


@app.route("/music", methods=['GET'])
def get_music_recs():
    # return jsonify({"songs": "its a win"}), 200
    emotion = request.args.get('emotion')
    # print(emotion)
    songs = recommend_music(emotion)
    return jsonify({"songs": songs}), 200


@app.route("/members")
def members():
    # return {"members":["Member1","Member2","Member3"]}
    try:
        with open('../emo.txt', 'r') as file:
            return file.read()
    except FileNotFoundError:
        return 'File not found', 404
if __name__=="__main__":
    app.run(debug=True,port=5000)
    
