from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration

# Load pre-trained BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Load pre-trained Seq2Seq model (T5 for text generation)
seq2seq_model = T5ForConditionalGeneration.from_pretrained("t5-small")

def generate_response(image_caption, user_query):
    """Generate chatbot responses using BERT + Seq2Seq"""
    
    # Encode user input with BERT
    input_text = f"Image: {image_caption}\nUser: {user_query}\nBot:"
    input_ids = bert_tokenizer.encode(input_text, return_tensors="pt")

    # Generate response using Seq2Seq model
    output = seq2seq_model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = bert_tokenizer.decode(output[0], skip_special_tokens=True)

    return response
