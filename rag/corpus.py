"""A tiny, self-contained knowledge base about a fictional product ("Orbit").

Self-contained so the eval cases have unambiguous ground truth and you can run
the whole suite without external data. Replace DOCUMENTS with your own chunks
(or load them from Chroma) once you wire this to real content.
"""

DOCUMENTS = [
    {
        "id": "plan-free",
        "source": "pricing.md",
        "text": "Orbit's Free plan includes 3 projects, 1 GB of storage, and "
        "community support. It does not include SSO or audit logs.",
    },
    {
        "id": "plan-pro",
        "source": "pricing.md",
        "text": "Orbit's Pro plan costs $20 per user per month, billed annually. "
        "It includes unlimited projects, 100 GB of storage, and email support.",
    },
    {
        "id": "plan-enterprise",
        "source": "pricing.md",
        "text": "Orbit's Enterprise plan adds SSO (SAML), audit logs, a 99.9% "
        "uptime SLA, and a dedicated account manager. Pricing is custom.",
    },
    {
        "id": "auth-sso",
        "source": "security.md",
        "text": "Single sign-on via SAML is available only on the Enterprise plan. "
        "Orbit supports Okta, Azure AD, and Google Workspace as identity providers.",
    },
    {
        "id": "data-region",
        "source": "security.md",
        "text": "Orbit stores customer data in the US by default. EU data residency "
        "(data stored in Frankfurt) is available on the Enterprise plan on request.",
    },
    {
        "id": "retention",
        "source": "security.md",
        "text": "Deleted projects are retained in backups for 30 days, after which "
        "they are permanently purged and cannot be recovered.",
    },
    {
        "id": "api-limits",
        "source": "api.md",
        "text": "The Orbit API allows 60 requests per minute on Free, 600 on Pro, "
        "and a configurable limit on Enterprise. Exceeding the limit returns HTTP 429.",
    },
    {
        "id": "support-sla",
        "source": "support.md",
        "text": "Support response targets are: best-effort on Free, next-business-day "
        "on Pro, and a 1-hour response for urgent issues on Enterprise.",
    },
]
