from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

def detectar_rachadura(imagem_path):
    imagem = cv2.imread(imagem_path)
    imagem = cv2.resize(imagem, (562, 426))
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_filtrada = cv2.GaussianBlur(imagem_gray, (5, 5), 0)
    bordas = cv2.Canny(imagem_filtrada, 50, 150)

    # Heurística simples: se tiver muitos contornos, considerar crítica
    risco = np.sum(bordas > 0) > 2000  # limiar simples

    # Salvar imagem processada temporariamente
    output_path = os.path.join(tempfile.gettempdir(), 'saida.png')
    cv2.imwrite(output_path, bordas)

    return risco, output_path

@app.route('/detectar', methods=['POST'])
def detectar():
    if 'imagem' not in request.files:
        return jsonify({'erro': 'Nenhuma imagem enviada'}), 400

    arquivo = request.files['imagem']
    temp_path = os.path.join(tempfile.gettempdir(), arquivo.filename)
    arquivo.save(temp_path)

    risco, saida_path = detectar_rachadura(temp_path)

    return jsonify({
        'risco': risco,
        'mensagem': 'Rachadura crítica' if risco else 'Rachadura leve',
        'imagem_processada': '/resultado'
    })

@app.route('/resultado', methods=['GET'])
def resultado():
    # Retorna a última imagem processada
    saida_path = os.path.join(tempfile.gettempdir(), 'saida.png')
    return send_file(saida_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
