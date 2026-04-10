"""Vercel serverless entrypoint for Flask app."""

from app import app

# Vercel Python runtime expects a module-level WSGI app named `app`.
