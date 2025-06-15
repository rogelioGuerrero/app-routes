import sys
import os

# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Intentando importar la aplicación...")
try:
    from main import app
    print("¡La importación de la aplicación fue exitosa!")
except Exception as e:
    print("--- ERROR DE IMPORTACIÓN DETECTADO ---")
    import traceback
    traceback.print_exc()
    print("-------------------------------------")
