import pandas as pd
import unicodedata
from pykml import parser
from os import path

"""
Script de utilidades para el proyecto de feminicidios
"""


# Normaliza un texto, si hay acentos o enyes, los quita
def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

# Normalize ALL DATA in a DF
def normalize_df(data):
    for column in data.columns:
        data[column] = data[column].apply(lambda x: normalize_text(x) if isinstance(x, str) else x)
        data[column] = data[column].str.lower()

    return data


# Las coordenas vienen como string, las convertimos a float
def DF_to_float(data):

    filtered_df = data.dropna(subset=['latitud', 'longitud'])

    filtered_df['latitud'] = pd.to_numeric(filtered_df['latitud'], errors='coerce')
    filtered_df['longitud'] = pd.to_numeric(filtered_df['longitud'], errors='coerce')
    final_data = filtered_df.dropna(subset=['latitud', 'longitud'])

    return final_data

# Regresa solo datos de Nuevo Leon
def DF_only_NL(data):

    # Como Nuevo Leon tiene acento, se normaliza el texto para que no haya problemas
    data['entidad'] = data['entidad'].apply(lambda x: normalize_text(x) if isinstance(x, str) else x)
    # Minusculas en entidad
    data['entidad'] = data['entidad'].str.lower()
    # Solo agarrar Nuevo Leon
    final_data = data[data['entidad'] == 'nuevo leon']

    return final_data

# PTM deberia de aprender geopandas en lugar de andar escribirndo parsers

def read_kml_to_dataframe(file_path):
    with open(file_path) as kml_file:
        doc = parser.parse(kml_file).getroot().Document

    data = []
    for pm in doc.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
        name = pm.name.text if hasattr(pm, 'name') else None
        description = pm.description.text if hasattr(pm, 'description') else None

        # Extract coordinates
        coords = pm.Point.coordinates.text.strip().split(',') if hasattr(pm, 'Point') else (None, None)
        longitude, latitude = coords[:2]

        data.append({
            'name': name,
            'description': description,
            'latitud': latitude,
            'longitud': longitude
        })

    return pd.DataFrame(data)