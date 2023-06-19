import pymongo
import random

# Create a MongoClient instance
client = pymongo.MongoClient("mongodb://localhost:27017/musicDB")

# Access the database and collection
db = client["musicDB"]
collection = db["music"]


def get_unique_emotions():
    seeds_list = db.collection.distinct("seeds")

    unique_words = set()
    emotions = set()
    for seed in seeds_list:

        words = seed.replace("[", "").replace("]", "").split(', ')
        unique_words.update(words)

    # print(unique_words)

    for mood in unique_words:
        mood_edited = mood.replace("'", "")
        emotions.add(mood_edited)
        # print(mood_edited)
    return emotions


def recommend_music(search_term):
    songs = list()
    # Query the collection
    results = collection.find({"seeds": {"$regex": search_term}})
    print(results)
    # Loop through the results and print the first 5
    result_list = list(results)
    stop = len(result_list)-5
    count = random.randrange(0, stop)
    loopend = count+5
    sub_result = result_list[count:loopend]
    for result in sub_result:
        songs.append(
            {'track': result['track'], 'artist': result['artist'], 'url': result['lastfm_url'],'spotify_id':result['spotify_id']})
    return songs
