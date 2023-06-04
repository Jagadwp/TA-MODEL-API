from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))


class Config:
    SECRET_KEY = environ.get("SECRET_KEY")
    FLASK_APP = environ.get("FLASK_APP")
    FLASK_DEBUG = environ.get("FLASK_DEBUG")
    HEWANI_DATASET_PATH = environ.get("HEWANI_DATASET_PATH")
    NABATI_DATASET_PATH = environ.get("NABATI_DATASET_PATH")