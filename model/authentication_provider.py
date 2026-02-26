import base64
import math
import os
import time
import uuid

import httpx
from aia_auth import auth


class AuthenticationProvider:
    def __init__(self) -> None:
        self.use_sso = os.getenv("USE_SSO", "false").lower() == "true"
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")

    def generate_auth_token(self) -> str:
        if self.use_sso:
            return self._sso()
        return self._get_bearer_token()

    def get_basic_credentials(self) -> str:
        self._validate_client_credentials()
        raw = f"{self.client_id}:{self.client_secret}".encode()
        return base64.b64encode(raw).decode()

    def _get_bearer_token(self) -> str:
        self._validate_client_credentials()
        return auth.client_credentials(self.client_id, self.client_secret).token

    def _sso(self) -> str:
        return auth.sso().token

    def _validate_client_credentials(self) -> None:
        if (
            self.client_id in (None, "", "Insert_your_client_id_here")
            or self.client_secret in (None, "", "Insert_your_client_secret_here")
        ):
            raise ValueError(
                "Invalid client credentials. Set CLIENT_ID and CLIENT_SECRET, "
                "or set USE_SSO=true for SSO mode."
            )


class AuthenticationProviderWithClientSideTokenRefresh(httpx.Auth):
    def __init__(self) -> None:
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.last_refreshed = math.floor(time.time())
        self.valid_until = math.floor(time.time()) - 1
        self.token = ""
        self.expires_in = 0

    def auth_flow(self, request):
        if "x-correlation-id" not in request.headers:
            request.headers["x-correlation-id"] = str(uuid.uuid4())
        request.headers["Authorization"] = f"Bearer {self.get_bearer_token()}"
        yield request

    def get_bearer_token(self) -> str:
        if self._is_expired():
            self.last_refreshed = math.floor(time.time())
            response = auth.client_credentials(self.client_id, self.client_secret)
            self.token = response.token
            self.expires_in = response.expires_in
            self.valid_until = self.last_refreshed + self.expires_in
        return self.token

    def _is_expired(self) -> bool:
        return time.time() >= self.valid_until
