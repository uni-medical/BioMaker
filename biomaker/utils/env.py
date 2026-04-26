def load_local_env() -> None:
    """Load local environment variables from .env when python-dotenv is installed."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
