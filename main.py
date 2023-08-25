from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
import cohere
from urllib.request import urlopen

### Load env variables ###
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

co = cohere.Client(f'{cohere_api_key}')

######

### Functions ###

def cosine_similarity(reference_embeddings, comparison_embeddings):

    reference_embeddings = np.array(reference_embeddings[0])
    comparison_embeddings = np.array(comparison_embeddings[0])

    cos_sim = dot(reference_embeddings, comparison_embeddings)/(norm(reference_embeddings)*norm(comparison_embeddings))

    return cos_sim

# load json file for deta
def load_json_file(url):

    response = urlopen(url)
      
    # returns JSON object as a dictionary
    data = json.load(response)

    return data

# Compare query text to full_text_embeddings

def calculate_cosine_sim(reference_embeddings, comparison_embeddings):
    '''
    Calculates the cosine similarity between reference and comparison embeddings

            Parameters:
                    reference_embeddings: The numpy array of the embeddings of the reference text
                    comparison_embeddings: The numpy array of the embeddings of the comparison text

            Returns:
                    cosine: A cosine similarity score
    '''

    cosine = cosine_similarity(reference_embeddings, comparison_embeddings)

    cosine = round(cosine,2)

    return cosine


def compare_to_segments(query_embeddings, all_segments_embeddings, extracted_texts, title):

  all_text_array = extracted_texts[title]

  output_array = []

  for index, item in enumerate(all_segments_embeddings):
    text = all_text_array[index][1]
    start_time = item[0]
    embeddings = item[1]

    cosine_similarity = calculate_cosine_sim(query_embeddings, embeddings)

    output_array.append([start_time, text, cosine_similarity])

  return output_array


def compare_to_full_text(query_text, videos_full_text_embeddings):

  cosine_score_array = []

  query_embeddings = co.embed(
      texts=[query_text],
      model='large',
  ).embeddings

  for key in videos_full_text_embeddings:
    embeddings = videos_full_text_embeddings[key]

    cosine_similarity = calculate_cosine_sim(query_embeddings, embeddings)

    cosine_score_array.append([key, cosine_similarity])

  return cosine_score_array, query_embeddings



def do_inference(query, top_k):

    cosine_score_array, query_embeddings = compare_to_full_text(query, videos_full_text_embeddings)

    # Rank scores
    scores = []

    for item in cosine_score_array:
      score = item[1]
      scores.append(score)

    top_k = -top_k

    top_k_idx = np.argsort(scores)[top_k:]
    top_k_idx = np.flip(top_k_idx)
    top_k_values = [scores[i] for i in top_k_idx]

    top_k_array = []

    for i in range(-top_k):
      current_index = top_k_idx[i]
      top_k_array.append([cosine_score_array[current_index][0], cosine_score_array[current_index][1]])

    # Get links
    for item in top_k_array:
      current_title = item[0]
      link = full_playlist_links[current_title]



    # Loop through the 5 videos
    start_timings_array = []

    for item_array in top_k_array:
      title = item_array[0]

      video_segments_array = videos_segments_embeddings[title]

      output_array = compare_to_segments(query_embeddings, video_segments_array, extracted_texts, title)

      # print(output_array)

      # Rank scores
      scores = []

      for item in output_array:
        score = item[2]
        scores.append(score)

      # print(scores)

      top_k_test = -top_k

      top_k_test_idx = np.argsort(scores)[top_k_test:]
      top_k_test_idx = np.flip(top_k_test_idx)
      top_k_test_values = [scores[i] for i in top_k_test_idx]

      top_k_test_array = []

      for i in range(-top_k_test):
        current_index = top_k_test_idx[i]
        top_k_test_array.append([output_array[current_index][0], output_array[current_index][1], output_array[current_index][2]])

      # # Uncomment this to check scores
      # for item in top_k_test_array:
      #   print(item)

      top_score_index = np.argsort(scores)[-1] # Sort in increasing order
      top_start_time = output_array[top_score_index][0]
      top_text = output_array[top_score_index][1]
      top_score = output_array[top_score_index][2]

      # print(top_start_time, top_text, top_score)

      start_timings_array.append([title ,top_start_time, top_text, top_score])

    # print(start_timings_array)

    print(f"Here are the top {str(top_k_test)} links with relevant start timings:")

    return_array = []

    # Get links
    for item in start_timings_array:
      current_title = item[0]
      link = full_playlist_links[current_title]
      current_timing = item[1]
      context = item[2]
      score = item[3]

      full_link = f"{link}&t={current_timing}"

      return_array.append([current_title, full_link, context, score])

      print(f"{current_title} - {link}&t={current_timing}")
      print(f"Context: {context}")
      print(f"Score: {score}")

    return return_array

print("Loading JSON files")

# Load json files
url_1 = "https://mendapararitik.github.io/SearchIT/extracted_texts.json"
url_2 = "https://mendapararitik.github.io/SearchIT/full_playlist_links.json"
url_3 = "https://mendapararitik.github.io/SearchIT/videos_full_text_embeddings.json"
url_4 = "https://mendapararitik.github.io/SearchIT/videos_segments_embeddings_1.json"
url_5 = "https://mendapararitik.github.io/SearchIT/videos_segments_embeddings_2.json"

extracted_texts = load_json_file(url_1)
full_playlist_links = load_json_file(url_2)
videos_full_text_embeddings = load_json_file(url_3)


videos_segments_embeddings = {**videos_segments_embeddings_1, **videos_segments_embeddings_2}

print("All JSON files loaded")

### Fast API code ###

app = FastAPI()

### For CORS ###

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##########


# Define the basemodel
class Item(BaseModel):
    query_string: str
    num_responses: int

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/query/")
async def create_item(query_string: Item):

    query = query_string.query_string
    num_responses = query_string.num_responses

    responses = do_inference(query, num_responses)

    print(responses)

    # Initialise empty dictionary
    json_response = {}

    for index, response in enumerate(responses):
        json_response[f'response {index+1}'] = response

    return json_response

