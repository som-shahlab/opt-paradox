# src/utils/logging.py 

import os
from rich.console import Console
from rich.panel import Panel

# Determine DEBUG status from the "DEBUG" environment variable
# If "DEBUG" is set to any non-empty string, DEBUG_FLAG will be True.
DEBUG_FLAG = bool(os.environ.get("DEBUG"))

# If not in DEBUG mode, set RICH_DISABLE to turn off Rich formatting globally
# This should be done before importing Console if it's to affect all instances
if not DEBUG_FLAG:
    os.environ["RICH_DISABLE"] = "1"

# Global console for standard output (prints to terminal)
# This is created when the module is imported and ready to use.
console = Console()

file_console = None

def initialize_file_logging(log_filename: str) -> bool:
    """
    Initializes the global `file_console` to write Rich output to the specified file.
    This function should be called once, typically at the start of the application if file logging is desired.

    Args:
        log_filename: The path to the log file.

    Returns:
        True if file logging was successfully initialized, False otherwise.
    """
    global file_console # We are assigning to the global variable
    try:
        # Open in 'w' mode to overwrite the log file on each run. 
        # Change to 'a' if you want to append to an existing log file.
        log_file_stream = open(log_filename, "w", encoding="utf-8")
        file_console = Console(file=log_file_stream, highlight=False, force_terminal=False, force_jupyter=False)
        # Adding a confirmation message to the main console:
        console.print(f":floppy_disk: File logging initialized. Outputting Rich logs to: [blue underline]{log_filename}[/blue underline]")
        return True
    except Exception as e:
        # Print error to the main console if file_console setup fails
        console.print(f"[bold red]:x: Error initializing file logging to '{log_filename}': {e}[/bold red]")
        file_console = None # Ensure file_console is None if initialization failed
        return False

def safe_print(text, title=""):
    """
    Prints text with a title, using Rich Panel if DEBUG_FLAG is True, 
    otherwise uses plain print.
    """
    if DEBUG_FLAG:
        # Use the global console for Panel output to terminal, it will be Rich if RICH_DISABLE is not active.
        console.print(Panel(str(text), title=str(title))) # Ensure text and title are strings
    else:
        # Use standard print for non-DEBUG to ensure it's truly plain
        # This will not use Rich, especially if RICH_DISABLE is set via os.environ
        print(f"[{str(title)}] {str(text)[:120]}…") # Ensure text and title are strings
