# Reinmaker API & Webhook Overview (Draft)

> Status: Draft – update once Reinmaker backend implementation begins.

## Purpose
- Document the initial contract between Lesson of the Day PBC core services and the Reinmaker game experience.
- Ensure shared assets (PhaseDNA, audio, animation) flow reliably into Reinmaker without duplication.
- Provide a baseline for security reviews, SDK generation, and automated contract tests.

## Environments
| Environment | Base URL | Notes |
|-------------|----------|-------|
| Local (mock) | `http://localhost:7040` | FastAPI/Express mock server used during integration tests |
| Staging | `https://staging.api.reinmaker.lessonoftheday.com` | Mirrors production schema; uses staging entitlements and asset buckets |
| Production | `https://api.reinmaker.com` | Live tenant; enforced rate limits |

## Authentication
- **Method**: OAuth 2.0 client credentials (short-lived access tokens, 15 min default).
- **Token issuer**: Lesson of the Day Identity service (`auth.lessonoftheday.com`).
- **Scopes**:
  - `reinmaker.content.write` – ingest lesson manifests and asset metadata.
  - `reinmaker.entitlement.read` – query user/product entitlements.
  - `reinmaker.telemetry.write` – push gameplay events.
- **Key rotation**: minimum every 90 days; managed via Secrets Manager. Never check tokens into source control.

## Planned REST Endpoints

| Endpoint | Method | Scope(s) | Summary |
|----------|--------|----------|---------|
| `/v1/content/manifests` | `POST` | `reinmaker.content.write` | Ingest a new or updated content manifest referencing shared PhaseDNA, audio, A2F assets |
| `/v1/content/manifests/{manifestId}` | `GET` | `reinmaker.content.write` | Retrieve manifest and validation status |
| `/v1/entitlements/{userId}` | `GET` | `reinmaker.entitlement.read` | Return active game entitlements mapped from the unified billing service |
| `/v1/telemetry/events` | `POST` | `reinmaker.telemetry.write` | Batch gameplay analytics events (<=500 events per request) |
| `/v1/health` | `GET` | none | Health check used by monitoring and CI |

### Manifest Submission Contract (Draft JSON Schema)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://api.reinmaker.com/v1/schemas/content-manifest.json",
  "type": "object",
  "required": ["manifestId", "version", "product", "assets", "source"],
  "properties": {
    "manifestId": {"type": "string"},
    "version": {"type": "string", "pattern": "^[0-9]{4}\.[0-9]{2}\.[0-9]{2}(?:-[a-z0-9]+)?$"},
    "product": {"enum": ["reinmaker", "daily_lesson", "curious_kelly"]},
    "source": {
      "type": "object",
      "required": ["phaseDNAId", "locale", "agePersona"],
      "properties": {
        "phaseDNAId": {"type": "string"},
        "locale": {"enum": ["en-US", "es-ES", "fr-FR"]},
        "agePersona": {"type": "string"}
      }
    },
    "assets": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "type", "url", "hash", "bytes"],
        "properties": {
          "id": {"type": "string"},
          "type": {"enum": ["overlay", "fx", "audio", "animation"]},
          "url": {"type": "string", "format": "uri"},
          "hash": {"type": "string", "pattern": "^[A-Fa-f0-9]{64}$"},
          "bytes": {"type": "integer", "minimum": 1},
          "metadata": {"type": "object"}
        }
      }
    }
  }
}
```

## Webhooks (Reinmaker → Core Services)

| Event | Payload (summary) | Target | Purpose |
|-------|-------------------|--------|---------|
| `reinmaker.entitlement.updated` | `{ userId, productTier, status, expiresAt }` | Billing service | Keep unified entitlements in sync |
| `reinmaker.content.requested` | `{ phaseDNAId, locale, agePersona }` | Content pipeline | Signal demand for assets not yet generated |
| `reinmaker.telemetry.flushed` | `{ batchId, events, processedAt }` | Analytics pipeline | Confirm ingestion and enable retries |

### Delivery & Retry Policy
- HTTPS POST with JSON body; signed using shared HMAC secret (rotated every 60 days).
- Expect `2xx` within 5 seconds. Otherwise, retry with exponential back-off: 30s, 2m, 10m, 1h, 6h (max 5 attempts).
- Dead-letter queue (SQS/EventHub) retains failed payloads for 14 days.

## Rate Limits (Draft)
- Authenticated clients: 120 requests/minute baseline per client ID.
- Burst capacity: 300 requests/minute for manifest ingestion; coordinate increases through DevOps.
- Webhook receivers must handle up to 5 concurrent deliveries.

## Security & Compliance Notes
- All requests must enforce TLS 1.3.
- Include `X-Lesson-Trace-Id` header for correlation across products.
- Log PII only in redacted form. Respect COPPA/FERPA obligations when Reinmaker targets child audiences.
- Conduct annual penetration testing and maintain SOC2 alignment.

## Next Steps
1. Flesh out OpenAPI spec once backend scaffolding starts.
2. Add Postman collection + contract tests using Prism/wiremock.
3. Align webhook payload schemas with analytics and billing teams.














