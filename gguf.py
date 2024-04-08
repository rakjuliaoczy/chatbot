from llama_cpp import Llama
LLM = Llama(model_path="mistral-7b-instruct-v0.1.Q5_K_M.gguf",verbose=False)
#prompt = "I bought an ice cream for 6 kids. Each cone was $1.25 and I paid with a $10 bill. How many dollars did I get back? Explain first before answering."


while(True):
    prompt = input('User: ')
    output = LLM(prompt,max_tokens=0,echo=False)
    print('Wiesiek: ', output["choices"][0]["text"])


# generate a response (takes several seconds)
# output = LLM(prompt,max_tokens=0,echo=False)
# print("-" * 80)
# print('Question: %s'%prompt)
# print("-" * 80)

# print('\n Answer:')
# print(output["choices"][0]["text"])