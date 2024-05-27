from flask import Flask, request, jsonify
from flask_cors import CORS
import string
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Define the vocabulary
vocabulary = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "
num_chars = len(vocabulary)

# Create mapping from characters to indices and vice versa
char_to_index = {char: idx for idx, char in enumerate(vocabulary)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Load the trained model
try:
    model = load_model('text_generation_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to generate text using the trained model
def generate_text(model, seed, n_chars, temperature=1.0):
    generated_text = seed
    for _ in range(n_chars):
        model_input = np.zeros((1, len(seed), num_chars))
        for t, char in enumerate(seed):
            if char in char_to_index:
                model_input[0, t, char_to_index[char]] = 1

        try:
            predictions = model.predict(model_input, verbose=0)[0]
        except Exception as e:
            print(f"Error during model prediction: {e}")
            break

        next_index = sample(predictions, temperature)  # Changed from predictions[-1] to predictions
        next_char = index_to_char[next_index]
        generated_text += next_char
        seed = seed[1:] + next_char

    return generated_text

# Sample function to sample from the predicted probability distribution
def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    seed = data.get('seed', '')
    n_chars = data.get('n_chars', 100)
    temperature = data.get('temperature', 0.5)

    try:
        generated_text = generate_text(model, seed, n_chars, temperature)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        print(f"Error in generate route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
