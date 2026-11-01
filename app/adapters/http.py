"""The outbound HTTP request layer: every page, feed and image fetch.

Every request goes through ``_get_following_redirects``, which SSRF-checks each
redirect hop before connecting.
"""

import io
import ipaddress
import socket
import time
import warnings
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
from requests.exceptions import RequestException

from app.domain.post import is_safe_url
from app.logger import logger

# Cap on redirects we will follow per request. Bounds redirect-chain abuse and
# keeps a misbehaving source from looping us.
MAX_REDIRECTS: int = 5


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if ``ip`` falls in an SSRF-sensitive range (private / loopback /
    link-local / reserved / multicast / unspecified) — notably the cloud
    metadata endpoint 169.254.169.254."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _is_blocked_host(host: str) -> bool:
    """True if ``host`` targets an SSRF-sensitive address (private / loopback /
    link-local / reserved range — notably the cloud metadata endpoint
    169.254.169.254).

    Two layers:
      1. IP-literal check (cheap): canonical dotted-quad / bracketed-IPv6, plus
         the decimal / hex / octal integer forms of an IPv4 address (e.g.
         ``2130706433``, ``0x7f000001``, ``017700000001`` all == 127.0.0.1) via
         ``inet_aton`` — the OS resolver accepts those, so without this they
         would slip past a plain ``ip_address`` parse.
      2. DNS resolution: a *hostname* is resolved and every returned A/AAAA
         address is checked, so a name that resolves to an internal address
         (e.g. attacker-controlled DNS pointing at 169.254.169.254) is blocked
         too. ``_get_following_redirects`` runs this on every hop.

    This is not a full DNS-rebinding defense (the address that ``requests``
    ultimately connects to could differ from the one resolved here), but it
    closes the practical gap: a source that links or redirects to an internal
    hostname.
    """
    if not host:
        return True
    try:
        return _is_blocked_ip(ipaddress.ip_address(host))
    except ValueError:
        pass
    # Not canonical IP text. Try the integer IPv4 encodings inet_aton accepts
    # (decimal/hex/octal) before treating it as a name to resolve.
    try:
        packed = socket.inet_aton(host)
    except OSError:
        pass
    else:
        return _is_blocked_ip(ipaddress.ip_address(packed))
    # A genuine hostname: resolve it and block if ANY resolved address is
    # sensitive. Resolution failure is treated as not-blocked here (the request
    # will simply fail to connect); only a positive hit on an internal address
    # blocks it.
    return _host_resolves_to_blocked(host)


def _host_resolves_to_blocked(host: str) -> bool:
    """Resolve ``host`` and return True if any resolved IP is SSRF-sensitive."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return False
    for info in infos:
        sockaddr = info[4]
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            return True
    return False


# A full, realistic browser header set. The Sec-Fetch-* / Sec-Ch-Ua and
# Upgrade-Insecure-Requests headers reduce 403s from anti-bot filters that
# reject bare User-Agent requests. One set: over 90 days of Batch logs, no
# second or third set ever rescued a request this one had lost.
REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}
REQUEST_TIMEOUT: int = 30


def parse_page(response: requests.Response) -> BeautifulSoup:
    """Parse a fetched page, decoding it with the charset the page declares.

    Without a charset in Content-Type, ``requests`` assumes ISO-8859-1 for text/*,
    and ``response.text`` turned every UTF-8 page from deepmind.google and
    sebastianraschka.com into mojibake ("Xiaomiâs") before the model saw it. Given
    bytes, BeautifulSoup reads the page's own <meta charset> instead.
    """
    if "charset=" in response.headers.get("content-type", "").lower():
        return BeautifulSoup(response.text, "html.parser")
    return BeautifulSoup(response.content, "html.parser")


def _get_following_redirects(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    timeout: float = REQUEST_TIMEOUT,
    stream: bool = False,
) -> requests.Response | None:
    """GET ``url``, following up to ``MAX_REDIRECTS`` redirects manually and
    SSRF-validating every hop's host, the first included, before connecting.
    Returns the final response, or None if any hop targets a blocked host or the
    chain is too long. Raises RequestException (via session.get) so the caller's
    retry logic still applies to transient errors.

    This is the only SSRF check on the request path, so callers need none of
    their own. ``headers`` is passed per request rather than written onto the
    session.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        host = urlparse(current).hostname or ""
        if _is_blocked_host(host):
            logger.warning("Refusing request to blocked host: '%s'", current)
            return None
        response = session.get(
            current,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
            verify=True,
            stream=stream,
        )
        if response.is_redirect and response.headers.get("location"):
            current = urljoin(current, response.headers["location"])
            continue
        return response
    logger.warning("Too many redirects for '%s'; refusing.", url)
    return None


def try_request(
    session: requests.Session, url: str, headers: dict[str, str]
) -> requests.Response | None:
    """GET with up to two retries for transient errors (timeout / 5xx / 429).
    Returns the response on success, or None on failure.

    ``headers`` is passed per request, never written onto the session, so what
    is sent is exactly what the caller declared.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Redirects are followed manually: allow_redirects=True would connect
            # to an internal redirect target before its host could be checked.
            response = _get_following_redirects(session, url, headers)
            if response is None:
                return None
            response.raise_for_status()
            return response
        except RequestException as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            transient = status in (429, 500, 502, 503, 504) or status is None
            if transient and attempt < max_attempts - 1:
                # Exponential backoff, or Retry-After (429) when present: an
                # immediate re-fire against a rate limiter just earns another 429.
                delay = _retry_delay(resp, attempt)
                logger.debug(
                    "Transient error for '%s' (%s); retry %d after %.1fs",
                    url,
                    status,
                    attempt + 1,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning("Request to '%s' failed: %s", url, e)
            return None
    return None


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    """Backoff seconds: honor a numeric Retry-After header, else exponential
    (1s, 2s, ...) plus a fixed 0.1s step per attempt. Not jitter: there is no
    randomness, and one weekly job has no herd of clients to spread out."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 30.0)
    base = float(2**attempt)
    step = 0.1 * (attempt + 1)
    return min(base + step, 30.0)


def make_robust_request(url: str) -> requests.Response | None:
    """GET an article or index page with browser headers, or None if it fails.

    SSRF-checked at every hop, the first included; transient errors retried
    (see ``try_request``).
    """
    if response := try_request(requests.Session(), url, REQUEST_HEADERS):
        return response
    logger.error("Request failed for '%s'", url)
    return None


# Image types every mail client displays. WebP and AVIF are not shown by
# Outlook for Windows, which renders the alt text instead; SVG is blocked by
# most clients. A server answering octet-stream gives no type to display.
EMAIL_IMAGE_TYPES: frozenset[str] = frozenset({"image/png", "image/jpeg", "image/gif"})


# One short attempt per image: a static CDN either answers at once or the image
# is not worth holding the run for (the summarizer also caps the total).
IMAGE_PROBE_TIMEOUT_SECONDS = 10
# Enough of the file for PNG, GIF and JPEG headers (a JPEG's size marker can sit
# behind tens of KB of EXIF data); the rest of the image is never downloaded.
IMAGE_HEADER_BYTES = 64 * 1024


def probe_email_image(url: str) -> int | None:
    """The pixel width of the image at ``url`` if a mail client will display
    it (0 when the width cannot be read), else None.

    Fetched, not guessed from the URL: CDN URLs often carry no extension, and
    the only proof an image loads is that it does. The request asks for the
    email-safe types, so a content-negotiating CDN answers with one of them if
    it can. Only the first bytes are read. SSRF-checked at every redirect hop
    like every other fetch.
    """
    if not is_safe_url(url):
        return None
    headers = REQUEST_HEADERS | {
        "Accept": "image/png,image/jpeg,image/gif;q=0.9,*/*;q=0.1"
    }
    try:
        response = _get_following_redirects(
            requests.Session(),
            url,
            headers,
            timeout=IMAGE_PROBE_TIMEOUT_SECONDS,
            stream=True,
        )
    except RequestException as e:
        logger.info("Image probe failed for '%s': %s", url, e)
        return None
    if response is None:
        return None
    with response:
        if not response.ok:
            return None
        content_type = response.headers.get("content-type", "").split(";")[0]
        if content_type.strip().lower() not in EMAIL_IMAGE_TYPES:
            return None
        return _image_width(response, time.monotonic() + IMAGE_PROBE_TIMEOUT_SECONDS)


def _image_width(response: requests.Response, deadline: float) -> int:
    """The width from the image's header bytes, or 0 if they do not say.

    Only the header is parsed (``.size``), never the pixels. A header claiming a
    huge image trips Pillow's decompression-bomb guard, which is still a
    displayable image of unknown width, not a reason to lose the card.
    """
    head = b""
    try:
        for chunk in response.iter_content(chunk_size=16 * 1024):
            head += chunk
            # The timeout bounds each read, not the total: a host that drips
            # bytes would otherwise hold the probe indefinitely.
            if len(head) >= IMAGE_HEADER_BYTES or time.monotonic() >= deadline:
                break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(head)) as image:
                return int(image.size[0])
    except (
        RequestException,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ):
        return 0
