"""Adapters for the production stores named in specification sections 5, 6 and 18.

Each adapter implements a protocol the platform already depends on, so swapping one in changes
no retrieval, authorization or workflow code. The deterministic in-process implementations stay
the default: these adapters are installed through the `stores` extra and constructed explicitly.

Authorization note (specification section 4): a retrieval database is never the authorization
boundary here. Every adapter sends the tenant filter to the store *and* re-applies the
authoritative `is_authorized` check in Python before a chunk leaves the adapter, so a store
that is misconfigured, stale or wrong cannot widen access on its own.
"""
