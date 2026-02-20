import os
import webbrowser
import requests
import uuid
import threading
from flask import Flask, request
from urllib.parse import urlencode
from dotenv import load_dotenv, set_key

class AccessTokenGenerator:
    def __init__(self):
        load_dotenv()
        self.env_file = ".env"
        
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.redirect_uri = os.getenv("REDIRECT_URI")
        
        self.auth_base_url = os.getenv("AUTH_LINK_URL")
        self.token_url = "https://auth.truelayer.com/connect/token"
        
        self.app = Flask(__name__)
        self._setup_routes()

    def get_auth_link(self):
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "scope": "info accounts balance cards transactions direct_debits standing_orders offline_access",
            "redirect_uri": self.redirect_uri,
            "providers": "uk-ob-all uk-oauth-all",
            "state": str(uuid.uuid4())
        }
        
        auth_url = f"{self.auth_base_url}?{urlencode(params)}"
        return auth_url

    def _setup_routes(self):
        @self.app.route('/callback')
        def callback():
            code = request.args.get('code')
            if code:
                response = self._handle_success(code)
                self.stop_server()
                return response
            return "Authorization failed.", 400

    def _handle_success(self, code):
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "code": code
        }

        response = requests.post(self.token_url, data=payload)
        res = response.json()

        if "access_token" in res:
            set_key(self.env_file, "TRUELAYER_ACCESS_TOKEN", res["access_token"])
            set_key(self.env_file, "TRUELAYER_REFRESH_TOKEN", res["refresh_token"])
            return "<h1>Success!</h1><p>Tokens saved. This server is shutting down...</p>"
        
        return f"Token Exchange Failed: {res}", 500

    def stop_server(self):
        shutdown_func = request.environ.get('werkzeug.server.shutdown')
        if shutdown_func:
            shutdown_func()
        else:
            threading.Timer(1, lambda: os._exit(0)).start()
    
    def regenerate_auth_token_using_refresh_token(self):
        refresh_token = os.getenv("TRUELAYER_REFRESH_TOKEN")
        if not refresh_token:
            return False
        
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token
        }

        response = requests.post(self.token_url, data=payload)
        res = response.json()

        if "access_token" in res:
            set_key(self.env_file, "TRUELAYER_ACCESS_TOKEN", res["access_token"])
            set_key(self.env_file, "TRUELAYER_REFRESH_TOKEN", res["refresh_token"])
            return True
        return False