# Skynet L99 dependency audit — 2026-07-28

Scope: exact server/desktop dependency pins needed to test the hardened
OpenJarvis HTTP boundary. This receipt contains no credentials.

## Decision

The upstream lock was not safe to install:

- `python-multipart==0.0.22` is below the fix for CVE-2026-40347;
- `starlette==0.52.1` is below the fix for CVE-2026-48710;
- the newest FastAPI release was inside the mandatory seven-day cooldown.

The accepted exact set is:

| Package | Version | Uploaded (UTC) | Wheel SHA-256 | PyPI vulns | OSV vulns |
| --- | --- | --- | --- | ---: | ---: |
| FastAPI | 0.139.2 | 2026-07-16 15:06:19 | `b9ad015a835173d59865e2f5d8296fbc2b317bf56a2ba1a5bfbdd03de2fd4b1c` | 0 | 0 |
| Starlette | 1.3.1 | 2026-06-12 09:23:10 | `c7372aae11c3c3f26a42df7bd626cec2f47d03483d261d369516a615a53714c6` | 0 | 0 |
| python-multipart | 0.0.32 | 2026-06-04 16:18:57 | `ff6d3f776f16878c894e52e107296ffc890e913c611b1a4ec6c44e2821fe2e23` | 0 | 0 |
| Uvicorn | 0.51.0 | 2026-07-08 10:59:04 | `5d38af6cd620f2ae3849fb44fd4879e0890aa1febe8d47eb355fb45d93fe6a5b` | 0 | 0 |
| Pydantic | 2.12.5 | 2025-11-26 15:11:44 | `e561593fccf61e8a20fc46dfc2dfe075b8be7d0188df33f221ad1f0139180f9d` | 0 | 0 |

The unified lock also needed compatible versions for optional extras. They are
not part of `--extra server`, but their lock changes were audited:

| Package | Version | Uploaded (UTC) | PyPI vulns | OSV vulns |
| --- | --- | --- | ---: | ---: |
| fastar | 0.11.0 | 2026-04-13 17:09:53 | 0 | 0 |
| httptools | 0.8.0 | 2026-05-25 22:16:50 | 0 | 0 |
| prometheus-fastapi-instrumentator | 8.0.2 | 2026-06-23 09:39:32 | 0 | 0 |

`prometheus-fastapi-instrumentator==8.1.0` was rejected because it was uploaded
on 2026-07-26, inside the cooldown. The VLLM extra pins 8.0.2 instead.

## Checks

- PyPI JSON metadata and wheel hashes checked for every exact version.
- OSV exact-version API queries returned zero findings.
- Socket package/version pages were checked for FastAPI, Starlette,
  python-multipart, Uvicorn, fastar, httptools and the Prometheus
  instrumentator. Socket's direct Pydantic page was not retrievable through the
  available client; PyPI and OSV checks succeeded.
- Recent supply-chain searches did not identify a package-specific compromise
  for the accepted versions.
- `uv lock --dry-run` resolved successfully.
- `uv lock --check` passed after the exact lock update.
- `uv sync --extra server --no-dev --dry-run` selected 64 exact packages and
  did not select the unrelated yanked `grpcio==1.78.1` present under disabled
  OpenHands/VLLM extras.

The dry-run and audit do not by themselves authorize a service start. Runtime
installation and activation remain separate security gates.
