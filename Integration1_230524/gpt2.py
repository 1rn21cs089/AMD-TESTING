from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"  # You can choose a different model size
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prompt for text generation
prompt_text = "Once upon a time, in a land far, far away"

# Tokenize input text
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

# Decode generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
