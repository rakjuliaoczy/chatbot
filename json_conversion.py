import json

# Manually extracted content
entries = [
    {
        "question": "Sounds nice! Where is it?",
        "answer": "It will be hosted at the local park in den Hague.",
        "context": "Bot: Sounds nice! Where is it? User: It will be hosted at the local park.",
        "topic": "location"
    },
    {
        "question": "I like this area. Are you looking for a DJ, band, ensemble or solo artist?",
        "answer": "I'm looking for a solo artist.",
        "context": "Bot: Sounds nice! Where is it? User: It will be hosted at the local park in den Hague. Bot: I like this area. Are you looking for a DJ, band, ensemble or solo artist? User: I'm looking for a solo artist.",
        "topic": "formation"
    },
    {
        "question": "Good choice. Which music genre are you interested in?",
        "answer": "I'm interested in classical music.",
        "context": "Bot: Sounds nice! Where is it? User: It will be hosted at the local park in den Hague. Bot: I like this area. Are you looking for a DJ, band, ensemble or solo artist? User: I'm looking for a solo artist. Bot: Good choice. Which music genre are you interested in? User: I'm interested in classical music.",
        "topic": "genre"
    },
    {
        "question": "Nice. How much would you like to pay?",
        "answer": "I have a budget of $800.",
        "context": "Bot: Sounds nice! Where is it? User: It will be hosted at the local park in den Hague. Bot: I like this area. Are you looking for a DJ, band, ensemble or solo artist? User: I'm looking for a solo artist. Bot: Good choice. Which music genre are you interested in? User: I'm interested in classical music. Bot: Nice. How much would you like to pay? User: I have a budget of $800.",
        "topic": "price"
    },
    {
        "question": "Noted. Check the website. Here are some artists that might interest you.",
        "answer": "",
        "context": "Bot: Sounds nice! Where is it? User: It will be hosted at the local park in den Hague. Bot: I like this area. Are you looking for a DJ, band, ensemble or solo artist? User: I'm looking for a solo artist. Bot: Good choice. Which music genre are you interested in? User: I'm interested in classical music. Bot: Nice. How much would you like to pay? User: I have a budget of $800. Bot: Noted. Check the website. Here are some artists that might interest you.",
        "topic": "conclusion"
    },

    {
        "question": "That sounds fantastic! Can you tell me the venue for this event?",
        "answer": "The event will take place in the community park in Amsterdam.",
        "context": "Bot: That sounds fantastic! Can you tell me the venue for this event? User: The event will take place in the community park.",
        "topic": "location"
    },
    {
        "question": "Great location! What kind of music performance are you envisioning? A DJ, a full band, a small group, or perhaps a solo performer?",
        "answer": "Actually, I'm thinking of hiring a solo performer.",
        "context": "Bot: That sounds fantastic! Can you tell me the venue for this event? User: The event will take place in the community park in Amsterdam. Bot: Great location! What kind of music performance are you envisioning? A DJ, a full band, a small group, or perhaps a solo performer? User: Actually, I'm thinking of hiring a solo performer.",
        "topic": "formation"
    },
    {
        "question": "Excellent choice. Do you have a particular music style in mind for this solo performer?",
        "answer": "Yes, I would prefer classical music for the event.",
        "context": "Bot: That sounds fantastic! Can you tell me the venue for this event? User: The event will take place in the community park in Amsterdam. Bot: Great location! What kind of music performance are you envisioning? A DJ, a full band, a small group, or perhaps a solo performer? User: Actually, I'm thinking of hiring a solo performer. Bot: Excellent choice. Do you have a particular music style in mind for this solo performer? User: Yes, I would prefer classical music for the event.",
        "topic": "genre"
    },
    {
        "question": "Sounds elegant. What's your budget for booking the artist?",
        "answer": "My budget is around $800.",
        "context": "Bot: That sounds fantastic! Can you tell me the venue for this event? User: The event will take place in the community park in Amsterdam. Bot: Great location! What kind of music performance are you envisioning? A DJ, a full band, a small group, or perhaps a solo performer? User: Actually, I'm thinking of hiring a solo performer. Bot: Excellent choice. Do you have a particular music style in mind for this solo performer? User: Yes, I would prefer classical music for the event. Bot: Sounds elegant. What's your budget for booking the artist? User: My budget is around $800.",
        "topic": "price"
    },
    {
        "question": "Understood. I've curated a selection that fits your criteria.",
        "answer": "",
        "context": "Bot: That sounds fantastic! Can you tell me the venue for this event? User: The event will take place in the community park in Amsterdam. Bot: Great location! What kind of music performance are you envisioning? A DJ, a full band, a small group, or perhaps a solo performer? User: Actually, I'm thinking of hiring a solo performer. Bot: Excellent choice. Do you have a particular music style in mind for this solo performer? User: Yes, I would prefer classical music for the event. Bot: Sounds elegant. What's your budget for booking the artist? User: My budget is around $800. Bot: Understood. I've curated a selection that fits your criteria.",
        "topic": "conclusion"
    },
    {
        "question": "Cool! Could you tell me where is it?",
        "answer": "Monachium",
        "context": "Bot: Cool! Could you tell me where is it? User: Monachium",
        "topic": "location"
    },
    {
        "question": "Sounds promising. Are you looking for a DJ, band, ensemble or solo artist?",
        "answer": "i dont care.",
        "context": "Bot: Cool! Could you tell me where is it? User: Monachium. Bot: Sounds promising. Are you looking for a DJ, band, ensemble or solo artist? User: i dont care.",
        "topic": "formation"
    },
    {
        "question": "Okay then. Which music genre are you interested in?",
        "answer": "maybe hip hop or rock",
        "context": "Bot: Cool! Could you tell me where is it? User: Monachium. Bot: Sounds promising. Are you looking for a DJ, band, ensemble or solo artist? User: i dont care. Bot: Okay then. Which music genre are you interested in? User: maybe hip hop or rock",
        "topic": "genre"
    },
    {
        "question": "Okay. How much would you like to pay?",
        "answer": "i dont wont to spend more than 1000 euros",
        "context": "Bot: Cool! Could you tell me where is it? User: Monachium. Bot: Sounds promising. Are you looking for a DJ, band, ensemble or solo artist? User: i dont care. Bot: Okay then. Which music genre are you interested in? User: maybe hip hop or rock. Bot: Okay. How much would you like to pay? User: i dont wont to spend more than 1000 euros",
        "topic": "price"
    },
    {
        "question": "Noted. Here are some artists that might interest you.",
        "answer": "",
        "context": "Bot: Cool! Could you tell me where is it? User: Monachium. Bot: Sounds promising. Are you looking for a DJ, band, ensemble or solo artist? User: i dont care. Bot: Okay then. Which music genre are you interested in? User: maybe hip hop or rock. Bot: Okay. How much would you like to pay? User: i dont wont to spend more than 1000 euros. Bot: Noted. Here are some artists that might interest you.",
        "topic": "conclusion"
    },
    {
        "question": "That sounds great! Can you share the location of the event?",
        "answer": "Monachium",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Monachium",
        "topic": "location"
    },
    {
        "question": "Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist?",
        "answer": "i dont care.",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Monachium. Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: i dont care.",
        "topic": "formation"
    },
    {
        "question": "Alright. Do you have a preference for the music genre? Hip hop, rock, or something else?",
        "answer": "maybe hip hop or rock",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Monachium. Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: i dont care. Bot: Alright. Do you have a preference for the music genre? Hip hop, rock, or something else? User: maybe hip hop or rock",
        "topic": "genre"
    },
    {
        "question": "Got it. What's your budget limit for this?",
        "answer": "i dont wont to spend more than 1000 euros",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Monachium. Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: i dont care. Bot: Alright. Do you have a preference for the music genre? Hip hop, rock, or something else? User: maybe hip hop or rock. Bot: Got it. What's your budget limit for this? User: i dont wont to spend more than 1000 euros",
        "topic": "price"
    },
    {
        "question": "Understood. I'll show you some artists that fit your criteria.",
        "answer": "",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Monachium. Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: i dont care. Bot: Alright. Do you have a preference for the music genre? Hip hop, rock, or something else? User: maybe hip hop or rock. Bot: Got it. What's your budget limit for this? User: i dont wont to spend more than 1000 euros. Bot: Understood. I'll show you some artists that fit your criteria.",
        "topic": "conclusion"
    },
    {
        "question": "Cool! Could you tell me where is it?",
        "answer": "Berlin",
        "context": "Bot: Cool! Could you tell me where is it? User: Berlin",
        "topic": "location"
    },
    {
        "question": "Sounds promising. Are you looking for a DJ, band, ensemble or solo artist?",
        "answer": "I'm not sure yet.",
        "context": "Bot: Cool! Could you tell me where is it? User: Berlin. Bot: Sounds promising. Are you looking for a DJ, band, ensemble or solo artist? User: I'm not sure yet.",
        "topic": "formation"
    },
    {
        "question": "Okay then. Which music genre are you interested in?",
        "answer": "Maybe electronic or indie.",
        "context": "Bot: Cool! Could you tell me where is it? User: Berlin. Bot: Sounds promising. Are you looking for a DJ, band, ensemble or solo artist? User: I'm not sure yet. Bot: Okay then. Which music genre are you interested in? User: Maybe electronic or indie.",
        "topic": "genre"
    },
    {
        "question": "Okay. How much would you like to pay?",
        "answer": "I'm flexible, but not more than $1000.",
        "context": "Bot: Cool! Could you tell me where is it? User: Berlin. Bot: Sounds promising. Are you looking for a DJ, band, ensemble or solo artist? User: I'm not sure yet. Bot: Okay then. Which music genre are you interested in? User: Maybe electronic or indie. Bot: Okay. How much would you like to pay? User: I'm flexible, but not more than $1000.",
        "topic": "price"
    },
    {
        "question": "Noted. Here are some artists that might interest you.",
        "answer": "",
        "context": "Bot: Cool! Could you tell me where is it? User: Berlin. Bot: Sounds promising. Are you looking for a DJ, band, ensemble or solo artist? User: I'm not sure yet. Bot: Okay then. Which music genre are you interested in? User: Maybe electronic or indie. Bot: Okay. How much would you like to pay? User: I'm flexible, but not more than $1000. Bot: Noted. Here are some artists that might interest you.",
        "topic": "conclusion"
    },
    {
        "question": "That sounds great! Can you share the location of the event?",
        "answer": "Miami",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami",
        "topic": "location"
    },
    {
        "question": "Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist?",
        "answer": "I'm not sure. It's for my sister's birthday and my father's 50th birthday celebration.",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami. Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: I'm not sure. It's for my sister's birthday and my father's 50th birthday celebration.",
        "topic": "formation"
    },
    {
        "question": "Alright. Do you have a preference for the music genre?",
        "answer": "I'm not picky, but something upbeat would be nice.",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami. Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: I'm not sure. It's for my sister's birthday and my father's 50th birthday celebration. Bot: Alright. Do you have a preference for the music genre? User: I'm not picky, but something upbeat would be nice.",
        "topic": "genre"
    },
    {
        "question": "Got it. What's your budget limit for this?",
        "answer": "I haven't decided yet, but I'll try to keep it reasonable.",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami. Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: I'm not sure. It's for my sister's birthday and my father's 50th birthday celebration. Bot: Alright. Do you have a preference for the music genre? User: I'm not picky, but something upbeat would be nice. Bot: Got it. What's your budget limit for this? User: I haven't decided yet, but I'll try to keep it reasonable.",
        "topic": "price"
    },
    {
        "question": "Understood. I'll keep that in mind while suggesting artists.",
        "answer": "",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami. Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: I'm not sure. It's for my sister's birthday and my father's 50th birthday celebration. Bot: Alright. Do you have a preference for the music genre? User: I'm not picky, but something upbeat would be nice. Bot: Got it. What's your budget limit for this? User: I haven't decided yet, but I'll try to keep it reasonable. Bot: Understood. I'll keep that in mind while suggesting artists.",
        "topic": "conclusion"
    },
    {
        "question": "That sounds great! Can you share the location of the event?",
        "answer": "Miami, I guess?",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami, I guess?",
        "topic": "location"
    },
    {
        "question": "Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist?",
        "answer": "Hmm, not sure, maybe just a DJ or something?",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami, I guess? Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: Hmm, not sure, maybe just a DJ or something?",
        "topic": "formation"
    },
    {
        "question": "Alright. Do you have a preference for the music genre?",
        "answer": "I don't know, something fun?",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami, I guess? Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: Hmm, not sure, maybe just a DJ or something? Bot: Alright. Do you have a preference for the music genre? User: I don't know, something fun?",
        "topic": "genre"
    },
    {
        "question": "Got it. What's your budget limit for this?",
        "answer": "Ehh, probably not too much, maybe like a few hundred bucks?",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami, I guess? Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: Hmm, not sure, maybe just a DJ or something? Bot: Alright. Do you have a preference for the music genre? User: I don't know, something fun? Bot: Got it. What's your budget limit for this? User: Ehh, probably not too much, maybe like a few hundred bucks?",
        "topic": "price"
    },
    {
        "question": "Understood. I'll keep that in mind while suggesting artists.",
        "answer": "",
        "context": "Bot: That sounds great! Can you share the location of the event? User: Miami, I guess? Bot: Interesting. What type of musical act are you considering? DJ, band, ensemble, or a solo artist? User: Hmm, not sure, maybe just a DJ or something? Bot: Alright. Do you have a preference for the music genre? User: I don't know, something fun? Bot: Got it. What's your budget limit for this? User: Ehh, probably not too much, maybe like a few hundred bucks? Bot: Understood. I'll keep that in mind while suggesting artists.",
        "topic": "conclusion"
    },  
    {
        "question": "Sounds nice! Where is it?",
        "answer": "In Berlin.",
        "context": "Bot: Sounds nice! Where is it? User: In Berlin.",
        "topic": "location"
    },
    {
        "question": "I like this area. Are you looking for a DJ, band, ensemble, or solo artist?",
        "answer": "Solo artist.",
        "context": "Bot: Sounds nice! Where is it? User: In Berlin. Bot: I like this area. Are you looking for a DJ, band, ensemble, or solo artist? User: Solo artist.",
        "topic": "formation"
    },
    {
        "question": "Good choice. Which music genre are you interested in?",
        "answer": "Pop.",
        "context": "Bot: Sounds nice! Where is it? User: In Berlin. Bot: I like this area. Are you looking for a DJ, band, ensemble, or solo artist? User: Solo artist. Bot: Good choice. Which music genre are you interested in? User: Pop.",
        "topic": "genre"
    },
    {
        "question": "Nice. How much would you like to pay?",
        "answer": "$600.",
        "context": "Bot: Sounds nice! Where is it? User: In Berlin. Bot: I like this area. Are you looking for a DJ, band, ensemble, or solo artist? User: Solo artist. Bot: Good choice. Which music genre are you interested in? User: Pop. Bot: Nice. How much would you like to pay? User: $600.",
        "topic": "price"
    },
    {
        "question": "Noted. Check the website. Here are some artists that might interest you.",
        "answer": "",
        "context": "Bot: Sounds nice! Where is it? User: In Berlin. Bot: I like this area. Are you looking for a DJ, band, ensemble, or solo artist? User: Solo artist. Bot: Good choice. Which music genre are you interested in? User: Pop. Bot: Nice. How much would you like to pay? User: $600. Bot: Noted. Check the website. Here are some artists that might interest you.",
        "topic": "conclusion"
    }
]

# Convert to JSON
output_json = json.dumps(entries, indent=4)

# Write JSON to file
with open('output.json', 'w') as json_file:
    json_file.write(output_json)