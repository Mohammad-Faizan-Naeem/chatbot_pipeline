

import faiss
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline ,AutoModelForCausalLM
import torch
from peft import PeftModel ,PeftConfig
from transformers import BitsAndBytesConfig
import os
from accelerate import dispatch_model

#load the customer service intent
with open("customer_intents.txt", "r") as file:
    customer_intents = set(line.strip().lower() for line in file)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

humen_df=pd.read_csv("humen_intents.csv")
quant_config = BitsAndBytesConfig(load_in_4bit=True)


device ="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#load the sentence transformer model
retrieve_model = SentenceTransformer("sentence_transformer_model").to(device)
instruction_embeddings = np.load("instruction_embeddings.npy")
response_embeddings = np.load("response_embeddings.npy")
faiss_index = faiss.read_index("instruction_index.faiss")


os.environ["WANDB_DISABLED"] = "true"


#load the responses for retrieval
cleaned_texts=pd.read_parquet("cleaned_texts.parquet")
responses = cleaned_texts["response"].tolist()


#load the intent classification model
intent_model=AutoModelForSequenceClassification.from_pretrained("intent_classification_model").to(device)
intent_tokenizer=AutoTokenizer.from_pretrained("intent_classification_model")


base_model_name = "meta-llama/Llama-3.2-1B"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    device_map={"": device},
    trust_remote_code=True
)
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
lora_model_path = "llama_finetuned_model"
config = PeftConfig.from_pretrained(lora_model_path)
llama_model = PeftModel.from_pretrained(base_model, lora_model_path, config=config , is_trainable=True)

#llama_model.print_trainable_parameters()

llama_tokenizer.pad_token = llama_tokenizer.eos_token

# ------------------------- INTENT CLASSIFICATION -------------------------


def predict_query_intent(query):
    with torch.no_grad():
        inputs = intent_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = intent_model(**inputs)
        intent_logits = outputs.logits
        intent_index = torch.argmax(intent_logits, dim=1).item()
    predicted_intent = label_encoder.inverse_transform([intent_index])[0]
    #print(f"Predicted Intent from predict_query_intent: {predicted_intent}")
    return predicted_intent





def classify_intent(query):

    predicted_intent = predict_query_intent(query)
    if predicted_intent in customer_intents:
        intent_category = "customer_service"
    elif predicted_intent in humen_df["intents"].tolist():
        intent_category = "human_interaction"
    else:
        intent_category = "unknown"
    #print(f"Classify Intent: {intent_category}")
    return intent_category


# ------------------------- RESPONSE RETRIEVAL -------------------------

def retrieve_response(query):
    query_embedding=retrieve_model.encode(query).reshape(1, -1).astype('float32')
    distance, indicies = faiss_index.search(query_embedding, 1)
    if indicies[0][0] == -1:
        print("No relevant response found in FAISS index!")
        return "I'm not sure how to answer that.", 1.0
    best_response_index = indicies[0][0]
    best_response = cleaned_texts.iloc[best_response_index]["response"]
    #print(f"Retrieved FAISS Response: {best_response} (Distance: {distance[0][0]:.2f})")
    return best_response, distance[0][0]

# ------------------------- RESPONSE GENERATION -------------------------
import re
def generate_response(query, max_length=150):

    """Generates a response using the fine-tuned LLaMA model."""
    formatted_query = f"{query}\nChatBot:"

    inputs = llama_tokenizer(formatted_query , return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    #print("Input Tensors to Model:", inputs)
    with torch.no_grad():
       output = llama_model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.7,
            pad_token_id=llama_tokenizer.eos_token_id
        )
    response = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.replace("ChatBot:", "").strip()

    if response.lower().startswith(query.lower()):
        response = response[len(query):].strip()


    response = re.split(r"Intent\s*:", response)[0].strip()
    return response


# ------------------------- CHATBOT PIPELINE -------------------------


def chatbot_pipeline(user_input):
    intent = classify_intent(user_input)
    if intent == "customer_service":
        response, distance = retrieve_response(user_input)
        print(f"Retrieved Response: {response}")
        if distance > 0.6:
            print("Low confidence, generating response...")
            response = generate_response(user_input)
    elif intent== "human_interaction":
        response = generate_response(user_input)
    else:
        response = "I'm not sure how to answer that."

        with open("unknown_queries.txt", "a") as file:
            file.write(user_input + "\n")

        print(f"Logged unknown query: {user_input}")
    return response

# ------------------------- MAIN PROGRAM -------------------------

if __name__ == "__main__":
     while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("ChatBot: Goodbye!")
            break
        print(chatbot_pipeline(user_input),"\n")

