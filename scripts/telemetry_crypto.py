#!/usr/bin/env python
"""
Encrypt the telemetry payload with a shared key.

WHY
===
The webhook endpoint is public: anyone holding the URL can read what arrives.
Encrypting the body removes that exposure entirely — the URL becomes useless
without the key, and the key is shared out of band and never enters the
repository.

This also means the endpoint does not need to be trusted. A public pastebin
would do.

WHAT IS USED
============
AES-256-GCM, via the `cryptography` package. GCM is authenticated encryption, so
a tampered or truncated payload fails to decrypt rather than silently decoding to
nonsense. The key is derived from a passphrase with PBKDF2-HMAC-SHA256.

Nothing here is hand-rolled. Rolling your own symmetric crypto is how data gets
silently corrupted or leaked, so this uses the standard library implementation
maintained by the Python Cryptographic Authority.

FORMAT
======
The body is a small self-describing envelope, so it can be decrypted with any
tool, not just this one:

    {
      "schema":      "proctor.metrics/1",
      "encrypted":   true,
      "alg":         "AES-256-GCM",
      "kdf":         "PBKDF2-HMAC-SHA256",
      "iterations":  200000,
      "salt":        "<base64, 16 bytes, unique per message>",
      "nonce":       "<base64, 12 bytes, unique per message>",
      "ciphertext":  "<base64>"
    }

The plaintext inside the ciphertext is exactly the payload that would otherwise
have been sent in the clear.

USAGE
=====
    # generate a strong key
    uv run python scripts/telemetry_crypto.py keygen

    # decrypt something you captured from the webhook
    uv run python scripts/telemetry_crypto.py decrypt captured.json

    # or from stdin
    curl -s "https://webhook.site/token/<uuid>/requests" | ... \
      | uv run python scripts/telemetry_crypto.py decrypt -

The key is read from PROCTOR_TELEMETRY_KEY, or prompted for.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import sys

ENV_KEY = "PROCTOR_TELEMETRY_KEY"

ALG = "AES-256-GCM"
KDF = "PBKDF2-HMAC-SHA256"
ITERATIONS = 200_000
KEY_BYTES = 32          # AES-256
SALT_BYTES = 16
NONCE_BYTES = 12        # 96-bit nonce, the GCM recommendation
ENVELOPE_SCHEMA = "proctor.metrics/1"


def generate_key(nbytes: int = 32) -> str:
    """A URL-safe random key. 32 bytes is ample entropy for PBKDF2 input."""
    return secrets.token_urlsafe(nbytes)


def _derive(key_material: str, salt: bytes, iterations: int) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=KEY_BYTES,
                      salt=salt, iterations=iterations
                      ).derive(key_material.encode("utf-8"))


def encrypt(payload: dict, key_material: str) -> dict:
    """dict -> self-describing encrypted envelope."""
    if not key_material:
        raise ValueError("no key material")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    key = _derive(key_material, salt, ITERATIONS)
    ct = AESGCM(key).encrypt(nonce, json.dumps(payload).encode("utf-8"), None)
    return {
        "schema": ENVELOPE_SCHEMA,
        "encrypted": True,
        "alg": ALG,
        "kdf": KDF,
        "iterations": ITERATIONS,
        "salt": base64.b64encode(salt).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ct).decode(),
    }


def is_envelope(obj) -> bool:
    return isinstance(obj, dict) and obj.get("encrypted") is True and "ciphertext" in obj


def decrypt(envelope: dict, key_material: str) -> dict:
    """Self-describing envelope -> the original payload."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not is_envelope(envelope):
        raise ValueError("not an encrypted envelope")
    if envelope.get("alg") != ALG:
        raise ValueError(f"unsupported algorithm {envelope.get('alg')!r}")
    if not key_material:
        raise ValueError("no key material")

    salt = base64.b64decode(envelope["salt"])
    nonce = base64.b64decode(envelope["nonce"])
    ct = base64.b64decode(envelope["ciphertext"])
    key = _derive(key_material, salt, int(envelope.get("iterations", ITERATIONS)))
    try:
        pt = AESGCM(key).decrypt(nonce, ct, None)
    except InvalidTag:
        raise ValueError("decryption failed - wrong key, or the payload was "
                         "altered in transit")
    return json.loads(pt.decode("utf-8"))


def resolve_key(explicit: str | None = None) -> str:
    return explicit or os.environ.get(ENV_KEY) or ""


# --------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("keygen", help="print a fresh key")

    d = sub.add_parser("decrypt", help="decrypt a captured envelope")
    d.add_argument("path", help="file containing the envelope, a webhook.site "
                                 "request, or the paginated API response; - for stdin")
    d.add_argument("--key", default="", help=f"or set {ENV_KEY}")
    d.add_argument("--index", type=int, default=0,
                   help="which request to use when the response holds several")

    e = sub.add_parser("encrypt", help="encrypt a JSON file, for testing")
    e.add_argument("path", help="file containing the payload, or - for stdin")
    e.add_argument("--key", default="")

    args = ap.parse_args()

    if args.cmd == "keygen":
        k = generate_key()
        print(k)
        print()
        print("  put it in .env (gitignored), and share it out of band:")
        print(f"  {ENV_KEY}={k}")
        print()
        print("  .env is not committed, so the key never enters the repository.")
        return 0

    if args.cmd == "encrypt":
        raw = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
        payload = json.loads(raw)
        key = resolve_key(args.key)
        if not key:
            print(f"error: no key. set {ENV_KEY} or pass --key", file=sys.stderr)
            return 1
        print(json.dumps(encrypt(payload, key), indent=2))
        return 0

    # decrypt
    raw = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    env = json.loads(raw)

    # Accept the three shapes this will realistically be handed:
    #   1. the bare encrypted envelope
    #   2. a single webhook.site request object  (has "content")
    #   3. the paginated API response            (has "data": [...])
    # Getting this wrong is easy: the requests API wraps everything in
    # {current_page, data: [...], total}, so piping it straight in needs the
    # unwrap below.
    if isinstance(env, dict) and isinstance(env.get("data"), list):
        if not env["data"]:
            print("error: that response contains no requests", file=sys.stderr)
            return 1
        idx = min(max(args.index, 0), len(env["data"]) - 1)
        if len(env["data"]) > 1:
            print(f"  note: {len(env['data'])} requests in the response, "
                  f"using index {idx} (newest first with sorting=newest); "
                  f"--index to choose another", file=sys.stderr)
        env = env["data"][idx]

    # webhook.site returns JSON bodies as raw text and base64-encodes anything
    # else, so try JSON first and fall back to base64.
    if isinstance(env, dict) and "content" in env and "ciphertext" not in env:
        body = env["content"]
        try:
            env = json.loads(body)
        except Exception:
            pad = "=" * ((4 - len(body) % 4) % 4)
            try:
                env = json.loads(base64.b64decode(body + pad).decode("utf-8"))
            except Exception as e:
                print(f"error: could not read the content field as JSON or base64 "
                      f"({type(e).__name__})", file=sys.stderr)
                return 1

    key = resolve_key(args.key)
    if not key:
        import getpass
        key = getpass.getpass(f"  {ENV_KEY}: ")

    try:
        payload = decrypt(env, key)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
