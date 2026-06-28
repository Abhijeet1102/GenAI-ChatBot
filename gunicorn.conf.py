import os

# Bind to 0.0.0.0 on the port specified by the PORT environment variable (defaulting to 10000 for Render)
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Increase the worker timeout to 120 seconds to allow the Hugging Face model
# (sentence-transformers/all-MiniLM-L6-v2) to download on startup without
# causing the worker to be killed.
timeout = 120

# Number of worker processes (Render's free tier has limited CPU/memory, 2 is usually safe)
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
