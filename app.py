from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from threading import Thread

# Crear la app de Flask
app = Flask(__name__)

# Habilitar CORS para todas las rutas
CORS(app)  # Esto habilita CORS para todos los orÃ­genes

# Cargar el modelo
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_name_2 = "Helsinki-NLP/opus-mt-en-es"
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_name_2)

# HTML como un string (lo puedes modificar)
html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prueba de Servicio de PredicciÃ³n</title>
    <script>
        async function sendPrediction() {
            const text = document.getElementById("inputText").value;
            const response = await fetch("http://localhost:5556/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            document.getElementById("predictionResult").innerText = `PredicciÃ³n: ${data.prediction}`;
        }
    </script>
</head>
<body>
    <div style="max-width: 600px; margin: 0 auto; text-align: center;">
        <h1>Servicio de PredicciÃ³n de Sentimiento</h1>
        <textarea id="inputText" rows="4" cols="50" placeholder="Escribe un texto para predecir su sentimiento..."></textarea>
        <br><br>
        <button onclick="sendPrediction()">Obtener PredicciÃ³n</button>
        <p id="predictionResult" style="margin-top: 20px; font-size: 18px; font-weight: bold;"></p>
    </div>
</body>
</html>
"""

html_content_translate = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Translation Service</title>
    <script>
        async function translateText() {
            const inputText = document.getElementById('input-text').value;

            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: inputText })
            });

            const data = await response.json();
            document.getElementById('output-text').innerText = data.translated_text;
        }
    </script>
</head>
<body>
    <div style="text-align: center; height: calc(100vh - 17px); display: flex; flex-direction: column;; justify-content: center; align-items: center;">
        <h1>Translate Text</h1>
        <textarea style="width: 600px; height: 100px" id="input-text" placeholder="Enter text to translate"></textarea>
        <br>
        <button onclick="translateText()">Translate</button>
        <br>
        <div id="output-text"></div>
    </div>
</body>
</html>
"""
@app.route("/")
def translate_front():
    return html_content_translate

# Ruta para servir el HTML
@app.route("/front/prediction")
def home():
    return html_content

# Ruta para la predicciÃ³n
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    inputs = tokenizer(data["text"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
    return jsonify({"prediction": prediction})

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text")

    inputs = tokenizer_2(text, return_tensors="pt")
    translated_ids = model_2.generate(**inputs)
    translated_text = tokenizer_2.batch_decode(translated_ids, skip_special_tokens=True)[0]

    return jsonify({"translated_text": translated_text})

# Ejecutar Flask en un hilo
def run_flask():
    app.run(host="0.0.0.0", port=5556, debug=True)

# Crear un hilo para ejecutar Flask
#thread = Thread(target=run_flask)
#thread.start()

if __name__ == '__main__':
    run_flask()