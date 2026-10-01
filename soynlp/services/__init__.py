"""Shared operations used by Python callers, CLI commands and YAML tasks."""

from .text import TEXT_METHODS, process_documents, text_processor

__all__ = ["TEXT_METHODS", "process_documents", "text_processor"]
