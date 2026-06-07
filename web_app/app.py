import os
from flask import Flask, render_template

app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/')
def chat():
    # Render the new chat UI and inject the API base URL used by the frontend JS.
    # The frontend uses `window.API_BASE || 'http://127.0.0.1:8000'` by default,
    # but setting it here helps when serving from Flask so same-origin requests
    # are possible during development.
    api_base = os.environ.get('BOUNCER_API_BASE', 'http://127.0.0.1:8000')
    return render_template('chat.html', api_base=api_base)


if __name__ == '__main__':
    # For local dev, run Flask on port 3000 so it doesn't conflict with the FastAPI server.
    app.run(host='127.0.0.1', port=3000, debug=True)
