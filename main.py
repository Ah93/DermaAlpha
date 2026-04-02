"""
main.py — DermaAlpha entry point.

Usage:
    python main.py
    python main.py --patient-id PT-ABC123
    python main.py --log-level DEBUG
"""

import argparse
import logging
import os
import sys

from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler for better error display
install_rich_traceback(show_locals=False)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DermaAlpha — AI Dermatology Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --patient-id PT-ABC123\n"
            "  python main.py --log-level DEBUG\n"
        ),
    )
    parser.add_argument(
        "--patient-id",
        metavar="ID",
        help="Pre-fill the patient ID (skips the ID prompt)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override the LOG_LEVEL from .env",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="DermaAlpha 1.0.0",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    # Suppress noisy third-party loggers unless debugging
    if numeric_level > logging.DEBUG:
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("pinecone").setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()

    # ── Load settings ─────────────────────────────────────────────────────────
    try:
        from config import ConfigurationError, settings
    except Exception as e:
        console.print(f"[bold red]Startup error:[/bold red] {e}")
        sys.exit(1)

    # Override log level from CLI if provided
    log_level = args.log_level or settings.log_level
    setup_logging(log_level)

    # ── Ensure image cache directory exists ───────────────────────────────────
    os.makedirs(settings.image_cache_dir, exist_ok=True)

    # ── Start the agent ───────────────────────────────────────────────────────
    try:
        from agent.derma_agent import DermaAgent
        from config import ConfigurationError

        agent = DermaAgent(settings)

        # If a patient ID was passed on the CLI, inject it into the first session
        if args.patient_id:
            console.print(
                f"[dim]Pre-filling patient ID: {args.patient_id.upper()}[/dim]\n"
            )
            # Monkey-patch the greeting to skip the ID prompt
            original_greeting = agent._greeting

            def patched_greeting(session):
                session.patient_id = args.patient_id.upper()
                session.is_new_patient = False
                from agent.patient_session import SessionState
                session.state = SessionState.COMPLAINT_INTAKE

            agent._greeting = patched_greeting  # type: ignore[method-assign]

        agent.run()

    except ConfigurationError as e:
        console.print(f"\n[bold red]Configuration error:[/bold red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Session interrupted. Goodbye.[/yellow]\n")
        sys.exit(0)
    except RuntimeError as e:
        console.print(f"\n[bold red]Runtime error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        logging.exception("Unhandled exception in main()")
        sys.exit(1)


if __name__ == "__main__":
    main()
