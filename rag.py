import torch
import os
import json
from openvino.runtime import Core
import argparse
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from sentence_transformers import SentenceTransformer
import streamlit as st  # Still need to import Streamlit for caching


# ANSI escape codes for colors
PINK = '\033[95m'  # Pink color for console output
CYAN = '\033[96m'  # Cyan color for console output
YELLOW = '\033[93m'  # Yellow color for console output
NEON_GREEN = '\033[92m'  # Neon green color for console output
RESET_COLOR = '\033[0m'  # Reset color to default

# Global variables for conversation history and vault content
conversation_history = []  # History of messages in the conversation
vault_embeddings_tensor = torch.tensor([])  # Tensor for embeddings, initially empty
vault_content = []  # Content of the vault (text documents)

system_message = """
You are a helpful assistant that is an expert at extracting the most useful information from a given text. 
System Prompt:

You are a nutrition expert with in-depth knowledge of healthy eating and food science. Your goal is to provide users with personalized, evidence-based food recommendations based on the food item they mention. You prioritize health, nutrient balance, and dietary guidelines in all your responses. For each food item, suggest healthy alternatives, preparation methods, or complementary foods to improve the nutritional value of the user's diet. Be concise, accurate, and clear in your recommendations.

Instructions:

    When a user inputs a specific food item (e.g., "pizza"), suggest healthier variations or alternatives, along with nutrient information and possible health benefits.
    Provide simple preparation methods if relevant (e.g., suggest how to make a healthier version of a dish).
    Consider common dietary preferences (e.g., vegetarian, vegan, low-carb, gluten-free) and adapt suggestions accordingly if the user specifies.
    If the food item is healthy, explain why and suggest complementary foods that could enhance the meal's nutritional profile."""

model_id = "OpenVINO/mistral-7b-instruct-v0.1-int8-ov"  # Model ID for OpenVINO
tokenizer = AutoTokenizer.from_pretrained(model_id)  # Load tokenizer for the model
model = OVModelForCausalLM.from_pretrained(model_id)  # Load the OpenVINO model

# Function to load an OpenVINO model from a specified path
def load_openvino_model(model_path):
    core = Core()  # Initialize OpenVINO Core
    model = core.read_model(model=model_path)  # Read the model
    compiled_model = core.compile_model(model, "MYRIAD")  # Compile the model for the MYRIAD device
    return compiled_model  # Return the compiled model

# Function to read the contents of a file and return it as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:  # Open file in read mode
        return infile.read()  # Return file contents

# Function to retrieve relevant context from the vault based on user input

def get_relevant_context(rewritten_input, vault_content, top_k=1):


    vault_embeddings = load_embeddings()

    if vault_embeddings is None or vault_embeddings.nelement() == 0:
        return []


    input_embedding = generate_embedding(rewritten_input)


    input_embedding_tensor = torch.tensor(input_embedding).unsqueeze(0)


    try:
        cos_scores = torch.cosine_similarity(input_embedding_tensor, vault_embeddings)


        top_k = min(top_k, cos_scores.size(0))
        top_indices = torch.topk(cos_scores, k=top_k)[
            1].tolist()
        relevant_context = [vault_content[idx].strip() for idx in top_indices]
    except Exception as e:
        relevant_context = []

    return relevant_context

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the embedding model for context generation

# Function to generate embeddings for the input text
def generate_embedding(text):
    embedding = embedding_model.encode(text)  # Generate embedding
    return embedding.tolist()  # Convert to list and return

# Function to rewrite the user query based on conversation history
def rewrite_query(user_input_json, conversation_history):
    user_input = json.loads(user_input_json)["Query"]  # Load the user query from JSON
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])  # Get recent context
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:

    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query

    Return ONLY the rewritten query text, without any additional formatting or explanations.

    Conversation History:
    {context}

    Original query: [{user_input}]

    Rewritten query: 
    """
    # Replace with actual logic to generate a rewritten query
    rewritten_query = "..."  # Placeholder for the rewritten query
    return json.dumps({"Rewritten Query": rewritten_query})  # Return as JSON

# Function to generate a response from the OpenVINO model
def generate_openvino_response(messages, tokenizer, model):
    # Solo incluir el último mensaje del usuario para obtener la respuesta adecuada
    user_message = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""

    # Prepare input for the model con el último mensaje del usuario
    inputs = tokenizer(user_message, return_tensors="pt")  # Tokeniza solo el mensaje del usuario

    # Generar una respuesta con los parámetros especificados
    outputs = model.generate(
        **inputs,
        max_length=500,  # Longitud máxima de la respuesta
        max_new_tokens=100,  # Limita el número de nuevos tokens generados
        do_sample=True,  # Permite generación aleatoria
        temperature=0.7,  # Controla la aleatoriedad en la generación
        top_k=50,  # Considera solo los 50 tokens más probables
        top_p=0.95  # Muestreo por núcleo
    )

    # Decodificar la salida y limpiar los espacios en blanco
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()  # Decodifica y elimina tokens especiales

    print("\033[91m" + response + "\033[0m")
    return response  # Return the generated response

# Main chat function to handle user input and generate responses
def chat(user_input, system_message, vault_embeddings, vault_content, conversation_history, tokenizer, model):
    conversation_history.append({"role": "user", "content": user_input})  # Append user input to history

    if len(conversation_history) > 1:  # If there is more than one message in history
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history)  # Rewrite the query
        rewritten_query_data = json.loads(rewritten_query_json)  # Load the rewritten query data
        rewritten_query = rewritten_query_data["Rewritten Query"]  # Get the rewritten query
        print(PINK + "Original Query: " + user_input + RESET_COLOR)  # Print original query
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)  # Print rewritten query
    else:
        rewritten_query = user_input  # If no history, use original input

    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)  # Get relevant context
    if relevant_context:
        context_str = "\n".join(relevant_context)  # Join relevant contexts into a string
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)  # Print context
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)  # Print if no context found

    user_input_with_context = user_input  # Initialize the user input with context
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str  # Add context to user input

    conversation_history[-1]["content"] = user_input_with_context  # Update the last message in history

    messages = [
        {"role": "system", "content": system_message},  # Add system message to messages
        *conversation_history  # Include the conversation history
    ]

    # Ensure tokenizer and model are passed to the function
    response = generate_openvino_response(messages, tokenizer, model)  # Generate the response
    conversation_history.append({"role": "assistant", "content": response})  # Append assistant's response to history

    return response  # Return the assistant's response

# Function to reset context when data source changes
def reset_context():
    global conversation_history, vault_embeddings_tensor, vault_content
    conversation_history = []  # Reset conversation history
    vault_content = []  # Reset vault content

    if os.path.exists("vault3.txt"):  # Check if the vault file exists
        with open("vault3.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()  # Load vault content from file

    print(NEON_GREEN + "Generating new embeddings for the vault content..." + RESET_COLOR)  # Notify about embedding generation
    vault_embeddings = []  # Initialize list for embeddings
    for content in vault_content:
        # Replace with your embedding generation logic
        embedding = generate_embedding(content)  # Generate embedding for each content
        vault_embeddings.append(embedding)  # Append embedding to the list

    global vault_embeddings_tensor
    vault_embeddings_tensor = torch.tensor(vault_embeddings)  # Convert list to tensor
    print("New embeddings generated and stored.")  # Notify that embeddings are generated



@st.cache_data
def embedding_processing():


    print(" Embedding processing starts")
    vault_embeddings = []
    if os.path.exists("vault3.txt"):
        with open("vault3.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()
    for content in vault_content:
        embedding = generate_embedding(content)  # Replace with actual logic
        vault_embeddings.append(embedding)
    # Convert to tensor and print embeddings
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    print("Embedding is done")


def user_chat(input_text):
    # Actualiza el historial de conversación solo si es necesario
    conversation_history.append({"role": "user", "content": input_text})

    response = chat(input_text, system_message, vault_embeddings_tensor, vault_content,
                    conversation_history, tokenizer, model)

    # Añade la respuesta al historial si lo necesitas
    conversation_history.append({"role": "assistant", "content": response})
    green_response = f"\033[92m{response}\033[0m"  # Aplica el color verde
    print(green_response)
    return response
embeddings_file = "vault_embeddings.pt"

def generate_embeddings_for_vault_content():
    global vault_embeddings_tensor, vault_content
    vault_embeddings = []
    if os.path.exists("vault3.txt"):
        with open("vault3.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()

    print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)

    for content in vault_content:
        embedding = embedding_model.encode(content)
        vault_embeddings.append(embedding)

    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    torch.save(vault_embeddings_tensor, embeddings_file)
    print("New embeddings generated and stored.")
    return vault_embeddings_tensor

def load_embeddings():
    try:

        vault_embeddings_tensor = torch.load('C:/Users/devcloud/PycharmProjects/pythonProject/MCBM/vault_embeddings.pt')
        return vault_embeddings_tensor
    except Exception as e:
        print(f"Error al cargar embeddings: {e}")
        return None


# # Parse command-line arguments
# print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
# parser = argparse.ArgumentParser(description="OpenVINO Chat")
# parser.add_argument("--model", default="OpenVINO/mistral-7b-instruct-v0.1-int8-ov",
#                     help="OpenVINO model to use (default: mistral)")
# args = parser.parse_args()

# # Load the vault content
# print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
# vault_content = []
# if os.path.exists("vault3.txt"):
#     with open("vault3.txt", "r", encoding='utf-8') as vault_file:
#         vault_content = vault_file.readlines()

# # Generate embeddings for the vault content using placeholder logic
# print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
# vault_embeddings = []
# for content in vault_content:
#     embedding = generate_embedding(content)  # Replace with actual logic
#     vault_embeddings.append(embedding)

# # Convert to tensor and print embeddings
# vault_embeddings_tensor = torch.tensor(vault_embeddings)
# print("Embeddings for each line in the vault:")
# print(vault_embeddings_tensor)


# # Conversation loop
# print("Starting conversation loop...")

# while True:
#     user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
#     if user_input.lower() == 'quit':
#         break

#     # Check if a new data source has been detected and reset context if needed
#     new_data_source_detected = False  # Set this to True when the data source changes
#     if new_data_source_detected:
#         reset_context()

#     # Suponiendo que ya tienes tokenizer y model definidos en tu código
#     response = chat(user_input, system_message, vault_embeddings_tensor, vault_content, conversation_history, tokenizer,
#                     model)
#     print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
