import pymongo

# Create a MongoClient instance
client = pymongo.MongoClient("mongodb://localhost:27017")

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
    count = 0
    for result in results:
        if count == 5:
            break
        songs.append(
            {'track': result['track'], 'artist': result['artist'], 'url': result['lastfm_url']})
        count += 1
    return songs
