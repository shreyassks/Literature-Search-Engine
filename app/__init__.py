# -*- coding: utf-8 -*-

from flask import Flask

app = Flask(__name__)


if app.config["ENV"] == "production":

    app.config.from_object("config.ProductionConfig")
    
else:
    app.config.from_object("config.DevelopmentConfig")
    

print("Environment in use :- ", str(app.config["ENV"]))

from app import main

