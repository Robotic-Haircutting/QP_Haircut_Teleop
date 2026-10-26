"""Create the self-signed HTTPS certificate used by the phone web page: certs/cert.pem and certs/key.pem.

Phone browsers only allow HTTPS pages to open the camera. The certificate covers this computer's host name and current LAN IP and is valid for 10 years.
The private key stays on this computer; certs/ is excluded in .gitignore. Never commit it.

Usage: python tools/make_cert.py [--ip 192.168.x.x ...]
"""
import argparse
import datetime
import ipaddress
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERTS = Path(__file__).resolve().parent.parent / "certs"


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", action="append", default=[], help="extra IP to include in the certificate; may be repeated")
    args = ap.parse_args()
    ips = {"127.0.0.1", *args.ip}
    if (ip := lan_ip()):
        ips.add(ip)
    host = socket.gethostname()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = x509.SubjectAlternativeName([x509.DNSName(host), x509.DNSName("localhost")]
                                      + [x509.IPAddress(ipaddress.ip_address(i)) for i in sorted(ips)])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=3650)).add_extension(san, critical=False)
            .sign(key, hashes.SHA256()))
    CERTS.mkdir(exist_ok=True)
    (CERTS / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                                      serialization.PrivateFormat.TraditionalOpenSSL,
                                                      serialization.NoEncryption()))
    (CERTS / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print("Created", CERTS / "cert.pem", "and", CERTS / "key.pem")
    print("Addresses:", host, "localhost", *sorted(ips))


if __name__ == "__main__":
    main()
