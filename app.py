from flask import Flask, request, jsonify, send_file, render_template
import mysql.connector
from flask_cors import CORS
import io
import os
import numpy as np
import json

# Importar OpenCV y scikit-image
import cv2
from skimage.metrics import structural_similarity as ssim

# Importar PyTorch y torchvision para la extracción de embeddings
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configurar dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo preentrenado MobileNetV2 y remover la capa de clasificación
model = models.mobilenet_v2(pretrained=True)
model.classifier = nn.Identity()  # Esto permite extraer las características (embedding)
model.to(device)
model.eval()

# Definir la transformación para procesar las imágenes
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)
CORS(app)

def connect_db():
    try:
        return mysql.connector.connect(
            host="192.168.1.21",
            user="emerson",
            password="shadow.2005.SHADOW",
            database="emerson"
        )
    except mysql.connector.Error:
        return None

def preprocess_with_opencv(image_bytes):
    """
    Utiliza OpenCV para preprocesar la imagen:
    - Convierte los bytes a un array de NumPy.
    - Decodifica la imagen en formato BGR.
    - Convierte a RGB.
    - Aplica un suavizado (Gaussian Blur) para reducir ruido.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    return image_rgb

def get_embedding_from_image_bytes(image_bytes, use_preprocessing=False):
    """
    Recibe los bytes de una imagen, la procesa y devuelve un vector de características (embedding)
    utilizando PyTorch. Si use_preprocessing es True, se aplica preprocesamiento con OpenCV.
    """
    if use_preprocessing:
        image_array = preprocess_with_opencv(image_bytes)
        image = Image.fromarray(image_array)
    else:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    image = transform_pipeline(image)
    image = image.unsqueeze(0).to(device)  # Añadir dimensión de batch y mover a dispositivo
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy().flatten()

def compare_images_ssim(image_bytes1, image_bytes2):
    """
    Compara dos imágenes utilizando el índice de similitud estructural (SSIM).
    Devuelve un score que indica el grado de similitud.
    """
    image1 = np.array(Image.open(io.BytesIO(image_bytes1)).convert("RGB"))
    image2 = np.array(Image.open(io.BytesIO(image_bytes2)).convert("RGB"))
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(image1_gray, image2_gray, full=True)
    return score

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_products", methods=["GET"])
def get_products():
    try:
        conn = connect_db()
        if not conn:
            return jsonify({"error": "Error de conexión a la base de datos"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, nombre, id_categoria, precio, id_marca, descripcion, cantidad FROM productos")
        productos = cursor.fetchall()
        for p in productos:
            p["imagen_url"] = f"http://localhost:5000/get_image/{p['id']}"
        cursor.close()
        conn.close()
        return jsonify(productos)
    except mysql.connector.Error as err:
        return jsonify({"error": f"Error en la base de datos: {err}"}), 500

@app.route("/get_image/<int:product_id>")
def get_image(product_id):
    try:
        conn = connect_db()
        if not conn:
            return "Error de conexión", 500
        cursor = conn.cursor()
        cursor.execute("SELECT imagen FROM productos WHERE id = %s", (product_id,))
        row = cursor.fetchone()
        if row and row[0]:
            return send_file(io.BytesIO(row[0]), mimetype="image/jpeg")
        return "Imagen no encontrada", 404
    except mysql.connector.Error as err:
        return f"Error al obtener la imagen: {err}", 500

@app.route("/add")
def mostrar_form():
    ruta_html = os.path.join(os.path.dirname(__file__), "add.html")
    return send_file(ruta_html)

@app.route("/add_product", methods=["POST"])
def add_product():
    data = request.form
    imagen = request.files.get("imagen")
    nombre = data.get("nombre", "")
    id_categoria = data.get("id_categoria", None)
    precio = data.get("precio", None)
    id_marca = data.get("id_marca", None)
    descripcion = data.get("descripcion", "")
    cantidad = data.get("cantidad", None)
    try:
        embedding = None
        if imagen:
            # Leer los bytes de la imagen y calcular el embedding.
            # Cambia use_preprocessing a True para usar el preprocesamiento con OpenCV.
            image_bytes = imagen.read()
            embedding_vector = get_embedding_from_image_bytes(image_bytes, use_preprocessing=True)
            embedding = json.dumps(embedding_vector.tolist())
            # Reiniciar el puntero para guardar la imagen en la BD
            imagen.seek(0)
        conn = connect_db()
        if not conn:
            return jsonify({"error": "Error de conexión"}), 500
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO productos
            (nombre, id_categoria, precio, id_marca, descripcion, cantidad, imagen, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            nombre,
            id_categoria,
            precio,
            id_marca,
            descripcion,
            cantidad,
            imagen.read() if imagen else None,
            embedding
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Producto agregado exitosamente"})
    except mysql.connector.Error as err:
        return jsonify({"error": f"Error en la base de datos: {err}"}), 500

@app.route("/categorias", methods=["GET"])
def get_categorias():
    try:
        conn = connect_db()
        if not conn:
            return jsonify({"error": "Error de conexión a la base de datos"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, nombre FROM categoria")
        categorias = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(categorias)
    except mysql.connector.Error as err:
        return jsonify({"error": f"Error en la base de datos: {err}"}), 500

@app.route("/marcas", methods=["GET"])
def get_marcas():
    try:
        conn = connect_db()
        if not conn:
            return jsonify({"error": "Error de conexión a la base de datos"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, nombre FROM marca")
        marcas = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(marcas)
    except mysql.connector.Error as err:
        return jsonify({"error": f"Error en la base de datos: {err}"}), 500

@app.route("/get_product/<int:product_id>", methods=["GET"])
def get_product(product_id):
    try:
        conn = connect_db()
        if not conn:
            return jsonify({"error": "Error de conexión a la base de datos"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, nombre, id_categoria, precio, id_marca, descripcion, cantidad
            FROM productos
            WHERE id = %s
        """, (product_id,))
        product = cursor.fetchone()
        cursor.close()
        conn.close()
        if product:
            product["imagen_url"] = f"http://localhost:5000/get_image/{product['id']}"
            return jsonify(product)
        else:
            return jsonify({"error": "Producto no encontrado"}), 404
    except mysql.connector.Error as err:
        return jsonify({"error": f"Error en la base de datos: {err}"}), 500

@app.route("/get_recommendations/<int:product_id>", methods=["GET"])
def get_recommendations(product_id):
    try:
        conn = connect_db()
        if not conn:
            return jsonify({"error": "Error de conexión a la base de datos"}), 500
        cursor = conn.cursor(dictionary=True)
        # Obtener el embedding y la categoría del producto seleccionado
        cursor.execute("""
            SELECT id, id_categoria, embedding
            FROM productos
            WHERE id = %s
        """, (product_id,))
        product = cursor.fetchone()
        if not product or not product["embedding"]:
            return jsonify({"error": "Producto no encontrado o sin embedding"}), 404

        selected_embedding = np.array(json.loads(product["embedding"]))
        selected_category = product["id_categoria"]

        # Recuperar productos de la misma categoría (excluyendo el seleccionado)
        cursor.execute("""
            SELECT id, nombre, id_categoria, precio, id_marca, descripcion, cantidad, imagen, embedding
            FROM productos
            WHERE id_categoria = %s AND id != %s
        """, (selected_category, product_id))
        products = cursor.fetchall()
        recommendations = []
        for p in products:
            if p["embedding"]:
                other_embedding = np.array(json.loads(p["embedding"]))
                cosine_sim = np.dot(selected_embedding, other_embedding) / (
                    np.linalg.norm(selected_embedding) * np.linalg.norm(other_embedding)
                )
                if cosine_sim >= 0.8:
                    p["similarity"] = float(cosine_sim)
                    p["imagen_url"] = f"http://localhost:5000/get_image/{p['id']}"
                    if "imagen" in p:
                        del p["imagen"]
                    recommendations.append(p)
        cursor.close()
        conn.close()
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare_images", methods=["POST"])
def compare_images():
    """
    Endpoint para comparar dos imágenes usando SSIM.
    Se espera recibir dos imágenes en la petición (imagen1 e imagen2).
    """
    try:
        imagen1 = request.files.get("imagen1")
        imagen2 = request.files.get("imagen2")
        if not imagen1 or not imagen2:
            return jsonify({"error": "Se requieren dos imágenes para comparar"}), 400
        image_bytes1 = imagen1.read()
        image_bytes2 = imagen2.read()
        score = compare_images_ssim(image_bytes1, image_bytes2)
        return jsonify({"ssim_score": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/add_category", methods=["POST"])
def add_category():
    data = request.form
    nombre_categoria = data.get("nombre_categoria", "")
    if not nombre_categoria:
        return jsonify({"error": "Falta el nombre de la categoría"}), 400
    try:
        conn = connect_db()
        if not conn:
            return jsonify({"error": "Error de conexión"}), 500
        cursor = conn.cursor()
        cursor.execute("INSERT INTO categoria (nombre) VALUES (%s)", (nombre_categoria,))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Categoría agregada exitosamente"})
    except mysql.connector.Error as err:
        return jsonify({"error": f"Error en la base de datos: {err}"}), 500

@app.route("/add_brand", methods=["POST"])
def add_brand():
    data = request.form
    nombre_marca = data.get("nombre_marca", "")
    if not nombre_marca:
        return jsonify({"error": "Falta el nombre de la marca"}), 400
    try:
        conn = connect_db()
        if not conn:
            return jsonify({"error": "Error de conexión"}), 500
        cursor = conn.cursor()
        cursor.execute("INSERT INTO marca (nombre) VALUES (%s)", (nombre_marca,))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Marca agregada exitosamente"})
    except mysql.connector.Error as err:
        return jsonify({"error": f"Error en la base de datos: {err}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
