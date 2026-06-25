from flask import Flask, request, render_template
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Cargar el pipeline entrenado
pipeline = joblib.load("modelo_avocado_pipeline.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("formulario.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Guardamos lo que el usuario envió para poder regresarlo al template
    # y que el formulario no se vacíe tras predecir.
    form_data = request.form

    try:
        # Obtener datos del formulario
        fecha_str = request.form["Date"]
        fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
        month = fecha.month
        year = fecha.year

        plu_4046   = float(request.form["4046"])
        plu_4225   = float(request.form["4225"])
        large_bags = float(request.form["Large Bags"])
        tipo       = request.form["type"]

        # Construir DataFrame con las columnas que espera el pipeline
        datos = pd.DataFrame([{
            "4046":       plu_4046,
            "4225":       plu_4225,
            "Large Bags": large_bags,
            "Month":      month,
            "year":       year,
            "type":       tipo,
        }])

        # Predecir
        prediccion = pipeline.predict(datos)[0]
        resultado  = f"${prediccion:.2f} USD"
        error      = None

    except Exception as e:
        resultado = None
        error     = f"Error al procesar la solicitud: {str(e)}"

    return render_template(
        "formulario.html",
        resultado=resultado,
        error=error,
        form_data=form_data
    )

if __name__ == "__main__":
    app.run(debug=True)