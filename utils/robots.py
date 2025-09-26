import urllib.robotparser as rp
from urllib.parse import urlparse

_RP_CACHE = {}

def robots_ok(url: str, user_agent: str = 'MediaLensBot/0.1') -> bool:
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    if root not in _RP_CACHE:
        r = rp.RobotFileParser()
        r.set_url(root)
        try:
            r.read()
        except Exception:
            # If robots.txt unreachable, default to True but you may choose False
            _RP_CACHE[root] = None
            return True
        _RP_CACHE[root] = r
    parser = _RP_CACHE[root]
    if parser is None:
        return True
    return parser.can_fetch(user_agent, url)
