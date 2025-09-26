from urllib.parse import urlparse, urlunparse

def canonicalize_url(url: str) -> str:
    """Basic canonicalization: lower-case scheme/host, strip fragments, remove default ports."""
    if not url:
        return url
    u = urlparse(url)
    scheme = (u.scheme or 'http').lower()
    netloc = u.hostname.lower() if u.hostname else ''
    if u.port and not ((scheme == 'http' and u.port == 80) or (scheme == 'https' and u.port == 443)):
        netloc = f"{netloc}:{u.port}"
    path = u.path or '/'
    query = '&'.join(sorted(filter(None, (u.query or '').split('&')))) or ''
    return urlunparse((scheme, netloc, path, '', query, ''))
