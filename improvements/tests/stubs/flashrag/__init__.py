"""Test-only stand-in for FlashRAG.

The frozen Judge integration tests exercise ``model.drag`` with fake inference
and retrieval, but that module still imports FlashRAG at module level. This
package supplies only those imports. It is placed on ``sys.path`` by the gate,
never by runtime code, and every constructor that would build a real model or
index raises instead.
"""
