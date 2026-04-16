"""Launch the Orbit2 Web Inspector."""

from src.inspector.web_inspector import serve

serve(".runtime/sessions.db")
