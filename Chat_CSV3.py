import random
import chainlit as cl
import asyncio
import pandas as pd
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import Request
from fastapi.responses import HTMLResponse
from chainlit.server import app
from chainlit.context import init_ws_context
from chainlit.session import WebsocketSession
#import vector_similarities

class RuleBasedChatBot:
    def __init__(self):
        self.conversation_state = "occasion"
        self.collected_data = {}

    def get_response(self, user_input):
        responses = {
            "occasion": "Sounds fun! Where is it?",
            "location": "Great! Are you looking for a DJ, band, ensemble, or solo artist?",
            "formation": "Interesting choice! Which music genre are you interested in?",
            "genre": "Noted. How much would you like to pay?",
            "budget": "Sure thing! I'll show you some artists that fit your criteria."
        }

        response = responses[self.conversation_state]
        self.collected_data[self.conversation_state] = user_input
        self.conversation_state = self.next_conversation_state()
        return response

    def next_conversation_state(self):
        states = ["occasion", "location", "formation", "genre", "budget"]
        current_index = states.index(self.conversation_state)
        return states[min(current_index + 1, len(states) - 1)]

bot = RuleBasedChatBot()

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "Snuppy"}
    return rename_dict.get(orig_author)

@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Snuppy",
        url="https://e7.pngegg.com/pngimages/96/827/png-clipart-club-penguin-clothing-avatar-penguin-blue-animals-thumbnail.png",
    ).send()
    await cl.Avatar(
        name="Human",
        url="https://duoplanet.com/wp-content/uploads/2023/05/duolingo-avatar-4.png",
    ).send()

    msg = cl.Message(content="Starting the bot...")
    await msg.send()

    greetings = ["Hi", "Hello", "Hey", "Hi there", "Hey there", "Hello there"]
    intros = ["I'm Snuppy, your personal assistant", "I'm Snuppy, here to assist you", "I'm Snuppy, at your service", "I'm Snuppy, excited to meet you"]
    help_statements = ["I hope I can help you find the best artist", "I'm here to assist you in finding the perfect artist", "I'm here to help you in finding the best artist", "Please let me know what you are looking for and I hope I can help you"]
    occasion_question = "What is the occasion you're searching for an artist for?"
    emojis = ["🎷", "🥁", "🎸"]

    greeting = random.choice(greetings)
    intro = random.choice(intros)
    help_statement = random.choice(help_statements)
    emoji = random.choice(emojis)

    msg_content = f"{greeting}! {emoji} {intro}. {help_statement}. {occasion_question}"
    msg.content = msg_content
    print("Session id:", cl.user_session.get("id"))
    await msg.update()

user_inputs = []

# @app.get("/")#main/{session_id}")
@cl.on_message
async def main(message):#, session_id: str):
    #ws_session = WebsocketSession.get_by_id(session_id=session_id)
    #init_ws_context(ws_session)
    global user_input
    user_input = message.content
    user_inputs.append(user_input)
    # print('user input:', user_input)
    
    # Check if the user input is "funeral"
    if user_input.lower() == "funeral" or user_input.lower() == "cremation":
        # If so, update the conversation state and respond accordingly
        bot.conversation_state = "location"
        response = "I'm sorry to hear that. What is the location?"
    else:
        # If not, proceed with the regular response
        response = bot.get_response(user_input)

    # if len(user_inputs) == 5:
    #     print(user_inputs)
    #     user_options = vector_similarities.all_options(user_inputs)
    #     print(user_options)

    await asyncio.sleep(1.3)
    await cl.Message(content=response).send()

# def get_user_inputs():
#     if len(user_inputs) == 5:
#         user_options = vector_similarities.all_options(user_inputs)
#         print(user_options)
#         return user_options



