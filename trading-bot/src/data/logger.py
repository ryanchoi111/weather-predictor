import httpx
import structlog


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if True else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


async def send_discord_alert(webhook_url: str | None, message: str, level: str = "info") -> None:
    if not webhook_url:
        return
    prefix = {"error": "\U0001f534", "warning": "\U0001f7e1", "info": "\U0001f7e2"}.get(level, "\u2139\ufe0f")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json={"content": f"{prefix} {message}"}, timeout=10)
    except Exception:
        pass
