"""Pluggable inference backends (LLM / ASR / TTS).

Real backends import their heavy dependencies lazily inside ``load()`` so that
mock mode runs with nothing but FastAPI installed.
"""
