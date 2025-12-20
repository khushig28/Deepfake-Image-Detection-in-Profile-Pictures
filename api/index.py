"""
Vercel Serverless Handler for Flask App
This file adapts the Flask app for Vercel's serverless environment
"""

from app import app

# Vercel serverless handler
def handler(request, response):
    return app(request, response)

# For Vercel
application = app
