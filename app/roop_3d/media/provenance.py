"""Hash-bound render recipes and optional, local cryptographic authentication."""
from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def capabilities() -> dict:
    return {'sha256': True, 'hmac-sha256': len(os.environ.get('ROOP3D_PROVENANCE_KEY', '').encode()) >= 32,
            'c2pa': bool(importlib.util.find_spec('c2pa') and os.environ.get('ROOP3D_C2PA_CERT_PEM')
                         and os.environ.get('ROOP3D_C2PA_KEY_PEM'))}


def require_mode(mode: str) -> None:
    if mode != 'none' and not capabilities().get(mode):
        raise DomainError('PROVENANCE_UNAVAILABLE', 'Requested provenance requires local SDK/credentials in environment variables', 422)


def recipe_claim(recipe: dict, mode: str) -> dict:
    require_mode(mode)
    result = {'schema': 'roop3d.render.v1', 'recipe_sha256': hashlib.sha256(canonical(recipe)).hexdigest(),
              'synthetic_media': True, 'authentication': mode}
    if mode == 'hmac-sha256':
        result['hmac'] = hmac.new(os.environ['ROOP3D_PROVENANCE_KEY'].encode(), canonical(result), hashlib.sha256).hexdigest()
    return result


def authenticate_manifest(manifest: dict, mode: str) -> None:
    if mode == 'hmac-sha256':
        manifest['authentication'] = {'algorithm': mode, 'value': hmac.new(
            os.environ['ROOP3D_PROVENANCE_KEY'].encode(), canonical(manifest), hashlib.sha256).hexdigest()}


def verify_manifest(media: Path, manifest: dict) -> dict:
    matches = hmac.compare_digest(sha256_file(media)[1], str(manifest.get('output_sha256', '')))
    authentication = manifest.get('authentication')
    authenticated = None
    if authentication and authentication.get('algorithm') == 'hmac-sha256':
        require_mode('hmac-sha256')
        content = {key: value for key, value in manifest.items() if key != 'authentication'}
        expected = hmac.new(os.environ['ROOP3D_PROVENANCE_KEY'].encode(), canonical(content), hashlib.sha256).hexdigest()
        authenticated = hmac.compare_digest(expected, str(authentication.get('value', '')))
    return {'media_hash_matches': matches, 'manifest_authenticated': authenticated,
            'valid': matches and authenticated is not False,
            'trust': 'shared local key' if authenticated else 'integrity only; no signer identity',
            'identity_proof': False,
            'identity_notice': "A SHA-256 hash establishes artifact byte integrity only; it does not prove a person's legal or biometric identity."}


def sign_c2pa(source: Path, destination: Path, claim: dict) -> dict:
    """Optional SDK, offline signer, no timestamp service or remote ingredients."""
    require_mode('c2pa')
    try:
        from c2pa import Builder, Reader, Context, Settings, Signer, C2paSigningAlg, C2paSignerInfo
        info = C2paSignerInfo(alg=C2paSigningAlg.ES256,
                             sign_cert=os.environ['ROOP3D_C2PA_CERT_PEM'].encode(),
                             private_key=os.environ['ROOP3D_C2PA_KEY_PEM'].encode(), ta_url=None)
        settings = Settings.from_dict({'verify': {'remote_manifest_fetch': False},
                                       'builder': {'thumbnail': {'enabled': False}}})
        definition = {'claim_generator': 'Roop3D/0.1', 'title': 'Locally processed synthetic media',
                      'assertions': [{'label': 'org.roop3d.render', 'data': claim},
                                     {'label': 'c2pa.actions', 'data': {'actions': [{'action': 'c2pa.edited',
                                       'digitalSourceType': 'http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia'}]}}]}
        with Context(settings) as context:
            with Signer.from_info(info) as signer, Builder(json.dumps(definition), context) as builder:
                builder.sign_file(str(source), str(destination), signer)
            with Reader(str(destination), context=context) as reader:
                report = json.loads(reader.json())
                active = report.get('active_manifest')
                if not active or active not in report.get('manifests', {}):
                    raise ValueError('Signed manifest missing')
                failures = report.get('validation_results', {}).get('activeManifest', {}).get('failure', [])
                failures += report.get('validation_status', [])
                if any(item.get('code') != 'signingCredential.untrusted' for item in failures):
                    raise ValueError('C2PA validation failed')
                assertions = report['manifests'][active].get('assertions', [])
                embedded = next((item.get('data') for item in assertions if item.get('label') == 'org.roop3d.render'), None)
                if embedded != claim:
                    raise ValueError('Render claim read-back differs')
                return {'embedded': True, 'active_manifest': active, 'trust': 'Verify certificate trust with your recipient'}
    except Exception:
        raise DomainError('C2PA_SIGN_FAILED', 'Local C2PA signing or read-back validation failed; check SDK and ES256 credentials', 422) from None
