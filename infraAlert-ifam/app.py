
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
app.secret_key = 'infraalert2025'
CORS(app, supports_credentials=True)

# Rota de login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if data['usuario'] == 'ifam' and data['senha'] == '1234':
        session['usuario'] = data['usuario']
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'erro', 'mensagem': 'Credenciais inválidas'}), 401

# Rota de detecção (protegida)
@app.route('/detectar', methods=['POST'])
def detectar():
    if 'usuario' not in session:
        return jsonify({'erro': 'Não autorizado'}), 401

    imagem = request.files['imagem']
    if not imagem:
        return jsonify({'erro': 'Nenhuma imagem enviada'}), 400

    # Processamento de imagem com OpenCV
    npimg = np.frombuffer(imagem.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (562, 426))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Salvar resultado temporariamente
    resultado_path = os.path.join(tempfile.gettempdir(), 'resultado.png')
    cv2.imwrite(resultado_path, edges)

    # Análise simples para simular risco
    risco = np.sum(edges > 0) > 3000
    mensagem = "Rachadura crítica detectada!" if risco else "Rachadura normal detectada."

    return jsonify({'risco': risco, 'mensagem': mensagem})

# Rota para retornar imagem processada
@app.route('/resultado')
def resultado():
    if 'usuario' not in session:
        return jsonify({'erro': 'Não autorizado'}), 401

    resultado_path = os.path.join(tempfile.gettempdir(), 'resultado.png')
    return send_file(resultado_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
