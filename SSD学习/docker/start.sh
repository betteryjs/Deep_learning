#!/bin/bash
sudo gunicorn -w 2 -b 0.0.0.0:5000 main:app
