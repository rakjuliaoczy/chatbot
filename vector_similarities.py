import Chat_CSV3
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from deep_translator import GoogleTranslator
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from word2number import w2n
import re

# user_inputs = Chat_CSV3.get_user_inputs()
# user_inputs = ['birthday party', 'Amsterdam', 'dj', 'pop', '200 euros']  
# print(user_inputs)

occasions = pd.read_csv('data_similarity_search/occasions.csv')
locations = pd.read_csv('data_similarity_search/locations.csv')
formations = pd.read_csv('data_similarity_search/formations.csv')
genres = pd.read_csv('data_similarity_search/genres.csv')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

translator = GoogleTranslator(source='auto', target='en')

occasions_list = list(set(occasions['name']))
locations_list = list(set(locations['city']))
formations_list = list(set(formations['name']))
genres_list = list(set(genres['name']))

budgets_list = ['All prices', '€0 to €200', '€201 to €400', '€401 to €600', '€601 to €1000', '€1001 +']

occasions_list = [translator.translate(occasion) for occasion in occasions_list]
locations_list = [translator.translate(location) for location in locations_list]
formations_list = [translator.translate(formation)  for formation in formations_list]
genres_list = [translator.translate(genre)  for genre in genres_list]

#user_inputs = ['birthday party', 'Amsterdam', 'dj', 'pop', '200 euros']

def words_to_options(user_inputs): #words_to_options(user_inputs) , user_inputs = ['birthday party', 'Amsterdam', 'dj', 'pop', '200 euros']     

    # Encode phrases and target phrase
    occasion_embeddings = model.encode(occasions_list, convert_to_tensor=True)
    location_embeddings = model.encode(locations_list, convert_to_tensor=True)
    formation_embeddings = model.encode(formations_list, convert_to_tensor=True)
    genre_embeddings = model.encode(genres_list, convert_to_tensor=True)

    target_embedding_occasion = model.encode(user_inputs[0], convert_to_tensor=True)
    target_embedding_location = model.encode(user_inputs[1], convert_to_tensor=True)
    target_embedding_formation = model.encode(user_inputs[2], convert_to_tensor=True)
    target_embedding_genre = model.encode(user_inputs[3], convert_to_tensor=True)

    # Compute cosine similarities
    similarities_occasion = util.pytorch_cos_sim(target_embedding_occasion, occasion_embeddings).squeeze().cpu().numpy()
    similarities_location = util.pytorch_cos_sim(target_embedding_location, location_embeddings).squeeze().cpu().numpy()
    similarities_formation = util.pytorch_cos_sim(target_embedding_formation, formation_embeddings).squeeze().cpu().numpy()
    similarities_genre = util.pytorch_cos_sim(target_embedding_genre, genre_embeddings).squeeze().cpu().numpy()

    # Find the index of the most similar phrase
    most_similar_index_occasion = similarities_occasion.argmax()
    most_similar_index_location = similarities_location.argmax()
    most_similar_index_formation = similarities_formation.argmax()
    most_similar_index_genre = similarities_genre.argmax()

    # Retrieve the most similar phrase
    most_similar_phrase_occasion = occasions_list[most_similar_index_occasion]
    most_similar_phrase_location = locations_list[most_similar_index_location]
    most_similar_phrase_formation = formations_list[most_similar_index_formation]
    most_similar_phrase_genre = genres_list[most_similar_index_genre]

    return most_similar_phrase_occasion, most_similar_phrase_location, most_similar_phrase_formation, most_similar_phrase_genre

    #Print all phrases with their similarity scores for debugging
    # for phrase, similarity in zip(phrases, similarities):
    #     print(f"Phrase: {phrase}, Similarity: {similarity:.4f}")


def extract_number_s(sentence):
    words = sentence.split()
    for word in words:
        try:
            number = w2n.word_to_num(word)
            return number
        except ValueError:
            continue
    return None

def extract_number(string):
    # Extract numeric part from the string
    number_match = re.search(r'\d+', string)
    if number_match:
        return int(number_match.group())

    # Try converting the entire string to a number
    try:
        number = w2n.word_to_num(string)
        return number
    except ValueError:
        # If conversion fails, return None
        return None


def get_budget_range(number):
    if number is None:
        return budgets_list[0]  # "All prices"
    elif number <= 200:
        return budgets_list[1]
    elif number <= 400:
        return budgets_list[2]
    elif number <= 600:
        return budgets_list[3]
    elif number <= 1000:
        return budgets_list[4]
    elif number > 1000:
        return budgets_list[5]
    else:
        return budgets_list[0]

#budget = 'doesnt matter'

def budget_to_option(budget): #budget = user_inputs[4] . user_inputs = ['birthday party', 'Amsterdam', 'dj', 'pop', '200 euros']     
    number = extract_number(budget)
    if number is None:  # If no number is found in digits
        number = extract_number_s(budget)  # Attempt to find a number in words

    budget_range = get_budget_range(number)
    return budget_range

def all_options(user_inputs):
    combined_output = list(words_to_options(user_inputs)) + [budget_to_option(user_inputs[4])]
    return combined_output

#print(all_options(user_inputs))
