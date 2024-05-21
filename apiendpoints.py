from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
import random
import asyncio
import chainlit as cl

# Import your chatbot class and instantiate it
from Chat_CSV3 import RuleBasedChatBot
bot = RuleBasedChatBot()

# Create a FastAPI instance
app = FastAPI()

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = bot.get_response(data)
        await websocket.send_text(response)

# Endpoint to start the conversation
@app.get("/start_conversation")
async def start_conversation():
    # Logic to start the conversation using the chatbot
    # For example, sending initial messages, avatars, etc.
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

    # Generate initial message for the user
    # You can customize this message as needed
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
    await msg.update()

# Endpoint to handle user messages
@app.post("/user_message")
async def user_message(request: Request):
    # Logic to handle user messages using the chatbot
    # For example, getting user input, processing it, and generating a response
    user_input = await request.json()
    response = bot.get_response(user_input)
    await asyncio.sleep(1.5)  # Simulate delay for typing
    return JSONResponse(content={"response": response})

# Endpoint to get user inputs
@app.get("/user_inputs")
async def get_user_inputs():
    # Logic to retrieve collected user inputs from the chatbot
    # For example, returning the collected data from the chatbot instance
    return bot.collected_data  # Assuming your chatbot collects user inputs

# Add more endpoints as needed

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
