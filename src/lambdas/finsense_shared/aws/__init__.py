"""Thin boto3 wrappers: one module per service, each defaulting to a lazily created client.

Every entry point takes an optional ``client`` so handlers and tests can inject a fake
instead of patching boto3.
"""
