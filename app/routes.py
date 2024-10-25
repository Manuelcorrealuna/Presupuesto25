from app import app

@app.route('/')
def index():
    return "Hola, la aplicación está funcionando!"