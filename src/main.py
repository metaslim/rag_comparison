"""
Main typer app for ConvFinQA
"""

import typer
from rich import print as rich_print

app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record"""
    history = []
    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        # TODO: YOUR CODE HERE
        response = "RESPONSE"
        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")
        history.append({"user": message, "assistant": response})


@app.command()
def myfunc() -> None:
    """My hello world function"""
    # TODO: YOUR CODE HERE
    rich_print("Hello World")


if __name__ == "__main__":
    app()
