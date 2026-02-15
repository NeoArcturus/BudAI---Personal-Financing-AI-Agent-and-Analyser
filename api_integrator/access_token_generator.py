import requests
import dotenv
import json
from flask import Flask, request, jsonify

payload = {

}


class AccessTokenGenerator:
    def __init__(self):
        self.access_token = dotenv.get("ACCESS_TOKEN")
        self.refresh_token = dotenv.get("REFRESH_TOKEN")
        self.client_id = dotenv.get("CLIENT_ID")
        self.client_secret = dotenv.get("CLIENT_SECRET")
        self.redirect_uri = dotenv.get("REDIRECT_URI")
        self.grant_type = dotenv.get("GRANT_TYPE")
        self.base_url = dotenv.get("BASE_URL")

    def generate_auth_link(self):
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "scope": "info accounts balance cards transactions direct_debits standing_orders offline_access",
            "redirect_uri": self.redirect_uri,
            "providers": "uk-ob-all uk-oauth-all"
        }

        p = requests.Request('GET', self.base_url, params=params).prepare()
        return p.url
