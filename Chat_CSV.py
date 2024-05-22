import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
from langchain_community.llms import CTransformers
import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from prompts_chat_pdf import chat_prompt, CONDENSE_QUESTION_PROMPT
import torch
from llama_cpp import Llama
import random
torch.device('cpu')
class CSVChatBot:

    def __init__(self):
        self.data_path = os.path.join('data')
        self.db_faiss_path = os.path.join('vectordb', 'db_faiss')
        #self.chat_prompt = PromptTemplate(template=chat_prompt, input_variables=['context', 'question'])
        #self.CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT

    def create_vector_db(self):

        '''function to create vector db provided the pdf files'''

        # Load data from CSV file
        # df = pd.read_csv(self.data_path)  # Assuming self.data_path contains the path to the CSV file

        # # Assuming the CSV file contains a column named 'text' that contains the text data
        # texts = df['text'].tolist()

        # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
        #                                 model_kwargs={'device': 'cpu'})

        # db = FAISS.from_documents(texts, embeddings)
        # db.save_local(self.db_faiss_path)

        # loader = DirectoryLoader(self.data_path,
        #                      glob='*.csv',
        #                      loader_cls=CSVLoader,
        #                      loader_kwargs={'encoding': 'utf-8'})

        # documents = loader.load()
        #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',documents)\
        jq_schema = {
            "question": ".question",
            "answer": ".answer",
            "context": ".context",
            "topic": ".topic"
        }

        loader = DirectoryLoader(self.data_path,
                             glob='*.csv',
                             loader_cls=CSVLoader)

        documents = loader.load()
        #print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,#400,
                                                   chunk_overlap=50)#0)
        texts = text_splitter.split_documents(documents)

        #embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       #model_kwargs={'device': 'cpu'})
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents(texts, embeddings)
        db.save_local("self.db_faiss_path")

    def load_llm(self):
        #Load the locally downloaded model here
        llm = CTransformers(
            #api_key="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",
            model="llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            max_new_tokens=2000,
            temperature=0.5,
            gpu_layers=32,
            device='cuda'
        )
        #llm = Llama(model_path="mistral-7b-instruct-v0.1.Q5_K_M.gguf",verbose=True)
        return llm

    def conversational_chain(self):

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # loader = DirectoryLoader(self.data_path,
        #                      glob='*.csv',
        #                      loader_cls=CSVLoader,
        #                      loader_kwargs={'encoding': 'utf-8'})
        
        # documents = loader.load()
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=20,#400,
        #                                            chunk_overlap=5)#0)
        # texts = text_splitter.split_documents(documents)
        # db = FAISS.from_documents(texts, embeddings)
        db = FAISS.load_local("self.db_faiss_path", embeddings, allow_dangerous_deserialization=True)
        # initializing the conversational chain

        #Design Prompt Template
        template = """You are a customer service chatbot for an online artist booking company called Gigstarter. You must provide answers about all the five topics: location (where it is), formation (DJ, band, ensemble, or solo artist?), genre type (music genre a customer is interested in), price (how much a customer is willing to pay)
        After location (where it is), you answer about formation (DJ, band, ensemble, or solo artist?). After formation you answer about genre type (Interesting choice! Which music genre are you interested in?). After genre type you answer about price (how much a customer is willing to pay). Provide each new answer with different topic SEPARATELY after receiving each user input. Finally, after responding about the price (how much a customer is willing to pay) say ONLY "Noted. I'll show you some artists that might interest you." or "Sure thing!. I'll show you some artists that fit your criteria.". 
        {context}

        Answer the customer's questions ONLY using the source data provided. Do NOT include in the response number of guests. If you are unsure, say "I don't know, please call our customer support". Use engaging, courteous, and professional language similar to a customer representative.
        Keep your answers concise. 

        Question: {question}

        Answer: """

        #Intiliaze prompt using prompt template via LangChain
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        print(
            prompt.format(
                context = "A customer is on the artist booking company website and wants to chat with the website chatbot to find the best artist match for them. The customer answers chatbot's questions. Chatbot starts conversation with: Hey! I'm Snuppy, excited to meet you. Please let me know what you are looking for and I hope I can help you. What is the occasion you're searching for an artist for?",
                question = "Human response",
                # input = "input"
                # answer = "User answers questions"
            )
        )

        chain_type_kwargs = {"prompt": prompt}

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversational_chain = ConversationalRetrievalChain.from_llm( llm=self.load_llm(),
                                                                      retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                                                                      combine_docs_chain_kwargs={"prompt": prompt},
                                                                      verbose=True,
                                                                      memory=memory,
                                                                      rephrase_question=False
                                                                      )

        return conversational_chain

def intialize_chain():
    bot = CSVChatBot()
    bot.create_vector_db()
    conversational_chain = bot.conversational_chain()
    return conversational_chain

chat_history = []

chain = intialize_chain()

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

    # Randomly select phrases
    greeting = random.choice(greetings)
    intro = random.choice(intros)
    help_statement = random.choice(help_statements)

    # Construct the message content
    msg_content = f"{greeting}! {intro}. {help_statement}. {occasion_question}"
    msg.content = msg_content
    await msg.update()

    cl.user_session.set("chain", chain)

# conversation_states = ["ocassion", "genre", "budget", "location", "complete"]

# # Initialize conversation state and collected data
# conversation_state = conversation_states[0]
# collected_data = {}


@cl.on_message
async def main(message):
    # global conversation_state
    # global collected_data
    global chat_history 

    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall({"question": message.content, "chat_history": chat_history}, callbacks=[cb])
    answer = res["answer"]
    chat_history.append(answer)
    #last_user_input = chat_history[-2] if chat_history else ""
    print('user message', message.content)
    print('answer',answer)
    #print('last_user_input', last_user_input)
    print('chat history', chat_history)
    #chat_history = chain.get_chat_history()
    # Update conversation state and collected data
    #collected_data[conversation_state] = message.content
    #print("Collected data dict: ", collected_data)
    #conversation_state = next_conversation_state(conversation_state)
    #print("Conversation state: ", conversation_state)

    # Determine the next question based on conversation state
    #next_question = determine_next_question(conversation_state)


    await cl.Message(content=answer).send()

# def next_conversation_state(current_state):
#     """Determine the next conversation state based on the current state."""
#     if current_state == "complete":
#         return "complete"
#     else:
#         return conversation_states[conversation_states.index(current_state) + 1]

# def determine_next_question(state):
#     """Determine the next question based on the conversation state."""
#     if state == "occasion":
#         return "Sounds fun! What genre would you like to listen to?"
#     elif state == "genre":
#         return "Sounds fun! What genre would you like to listen to?"
#     elif state == "budget":
#         return "Interesting. How much would you like to pay?"
#     elif state == "location":
#         return "Noted. Where is your event happening?"
#     elif state == "complete":
#         return "Okay. Thank you for providing the information. Based on what you provided I would recommend you the following artists."
    
# while(True):
#     query = input('User: ')
#     response = chain({"question": query, "chat_history": chat_history})
#     chat_history.append(response["answer"])  # Append the answer to chat history
#     print(response["answer"])