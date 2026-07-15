import os
import certifi
import ssl

def apply_ssl_fix():
    # Ultimate Nuke: Disable SSL verification globally at the core Python `ssl` module level
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    cert_path = certifi.where()
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = cert_path
    
    # Global SSL bypass for standard library
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Global SSL bypass for httpx (used by Google Gemini SDK)
    try:
        import httpx
        original_async_init = httpx.AsyncClient.__init__
        def new_async_init(self, *args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = httpx.Timeout(600.0) # 10 minute timeout!
            original_async_init(self, *args, **kwargs)
        httpx.AsyncClient.__init__ = new_async_init
        
        original_sync_init = httpx.Client.__init__
        def new_sync_init(self, *args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = httpx.Timeout(600.0) # 10 minute timeout!
            original_sync_init(self, *args, **kwargs)
        httpx.Client.__init__ = new_sync_init
    except ImportError:
        pass

    # Global SSL bypass for requests (used by Google API Core)
    try:
        import requests
        original_request = requests.Session.request
        def new_request(self, method, url, **kwargs):
            kwargs['verify'] = False
            return original_request(self, method, url, **kwargs)
        requests.Session.request = new_request
    except ImportError:
        pass
