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

## Follow-up: base/server transitive findings

The frozen environment scan also reported seven packages selected by the
`server` extra. These were not seven independent advisory records: OSV returned
55 records, of which 27 were duplicate PYSEC/GHSA aliases for the same CVE.
After grouping by CVE there were 28 distinct vulnerabilities:

| Package | Old | OSV records | Unique CVEs | Active server path | Accepted exact version |
| --- | --- | ---: | ---: | --- | --- |
| aiohttp | 3.13.3 | 42 | 21 | `datasets[http] -> fsspec -> aiohttp` | 3.14.1 |
| click | 8.3.1 | 1 | 1 | direct OpenJarvis dependency | 8.3.3 |
| idna | 3.11 | 2 | 1 | `httpx` / `requests` / `yarl` | 3.15 |
| lxml | 6.0.2 | 2 | 1 | `ddgs -> lxml` | 6.1.0 |
| pygments | 2.19.2 | 2 | 1 | `rich -> pygments` | 2.20.0 |
| requests | 2.32.5 | 2 | 1 | `datasets` / `posthog` | 2.33.0 |
| urllib3 | 2.6.3 | 4 | 2 | `requests -> urllib3` | 2.7.0 |

There were therefore no package-level false positives in this set. Optional
extras add more reverse paths, but `uv sync --extra server --no-dev --dry-run`
selected all seven through the base/server graph.

The accepted releases are the smallest later releases for which the exact
OSV-version query returned no findings:

| Package | Version | Uploaded (UTC) | PyPI vulns | OSV vulns | sdist SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| aiohttp | 3.14.1 | 2026-06-07 21:05:37 | 0 | 0 | `307f2cff90a764d329e77040603fa032db89c5c24fdad50c4c15334cba744035` |
| click | 8.3.3 | 2026-04-22 15:11:25 | 0 | 0 | `398329ad4837b2ff7cbe1dd166a4c0f8900c3ca3a218de04466f38f6497f18a2` |
| idna | 3.15 | 2026-05-12 22:45:55 | 0 | 0 | `ca962446ea538f7092a95e057da437618e886f4d349216d2b1e294abfdb65fdc` |
| lxml | 6.1.0 | 2026-04-18 04:27:24 | 0 | 0 | `bfd57d8008c4965709a919c3e9a98f76c2c7cb319086b3d26858250620023b13` |
| Pygments | 2.20.0 | 2026-03-29 13:29:30 | 0 | 0 | `6757cd03768053ff99f3039c1a36d6c0aa0b263438fcab17520b30a303a82b5f` |
| Requests | 2.33.0 | 2026-03-25 15:10:40 | 0 | 0 | `c7ebc5e8b0f21837386ad0e1c8fe8b829fa5f544d8df3b2253bff14ef29d7652` |
| urllib3 | 2.7.0 | 2026-05-07 16:13:17 | 0 | 0 | `231e0ec3b63ceb14667c67be60f2f2c40a518cb38b03af60abc813da26505f4c` |

All seven releases were more than seven days old on 2026-07-28. Exact-version
Socket pages were requested for every candidate; the direct client was denied
with HTTP 403. Socket's indexed package analysis was available for aiohttp
3.14.1 and urllib3 2.7.0, with package/version listings available for Click and
Pygments. PyPI metadata, OSV exact-version queries, official repository release
or changelog pages, and recent package-specific compromise searches were used
as the documented fallback; none identified a compromise of an accepted
release.

`click` is exact-pinned as a direct dependency. The six transitive packages are
exact-pinned in `[tool.uv].constraint-dependencies`, which restricts an existing
dependency without causing an otherwise-unused package to be installed.

The update sequence was:

- exact candidate resolution with `uv lock --dry-run -P package==version`;
- manifest constraints plus a second plain `uv lock --dry-run`;
- `uv lock`, preserving the existing lockfile and its artifact hashes;
- `uv lock --check`;
- `uv sync --extra server --no-dev --dry-run`, selecting 64 packages and all
  seven corrected versions without creating `.venv`;
- a final OSV querybatch, which returned zero findings for all seven exact
  versions;
- an OSV querybatch generated from the complete exported server resolution
  (70 platform-marked requirement records for the 64 selected packages), which
  returned zero vulnerable records.

The unified lock still warns about the pre-existing yanked
`grpcio==1.78.1` under disabled optional graphs and the pre-existing missing
`zeus-ml` `apple` extra. Neither package is selected by the 64-package
`--extra server --no-dev` dry-run, and neither was changed in this follow-up.
