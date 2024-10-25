from flask import Flask, render_template
import pandas as pd

def create_app():
    app = Flask(__name__)

    # Cargar los datos una vez
    gastos = pd.read_csv('data/Gastos.csv')
    recursos_humanos = pd.read_csv('data/RecursosHumanos.csv')

    @app.route('/')
    def index():
        # Llama a la plantilla HTML principal (index.html)
        return render_template('index.html')

    return app
