from dotenv import load_dotenv, find_dotenv
import os
env_path ='.env'
load_dotenv(dotenv_path=env_path, verbose=True)


class Config(object):
    DEBUG = False
    TESTING = False
    # Other global settings

class DevelopmentConfig(Config):
    FLASK_ENV = 'development'
    DEBUG = True
    # Development-specific settings

class ProductionConfig(Config):
    # Production-specific settings
    FLASK_ENV = 'production'
    CPU_LIMIT = os.getenv("cpu_limit")

class TestingConfig(Config):
    FLASK_ENV = 'test'
    TESTING = True
    # Testing-specific settings
