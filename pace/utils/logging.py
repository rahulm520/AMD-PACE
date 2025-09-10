# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import traceback
from enum import Enum
from typing import Optional
from contextlib import contextmanager

import pace


# Define the log levels
class logLevel(Enum):
    DEBUG = 0
    PROFILE = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


class PACELogger:

    # A flag to suppress logging
    supressor = False

    @classmethod
    def pacelogger(
        cls, loglevel: logLevel, message: str, caller_stack_depth: Optional[int] = 0
    ):

        def get_caller_info(caller_stack_depth: Optional[int] = 0):
            """
            Get the filename and line number of the caller of the function that calls this function.

            The limit is set to 3 + caller_stack_depth to get the caller info of the function that calls this function.
            3 is since the stack is 3 levels deep: get_caller_info -> pacelogger -> caller function.

            Args:
                caller_stack_depth (int): The number of stack frames to go back to get the caller info. Default is 0.
            """
            caller_stack = traceback.extract_stack(limit=3 + caller_stack_depth)[0]
            return os.path.basename(caller_stack.filename), caller_stack.lineno

        if cls.supressor:
            return

        assert isinstance(  # noqa: F631
            loglevel, logLevel
        ), f"loglevel must be of type logLevel, but got {type(loglevel)}"

        caller_info = get_caller_info(caller_stack_depth=caller_stack_depth)
        log_message = f"{caller_info[0]}:{caller_info[1]} {message}"
        pace.core.pace_logger(loglevel.value, log_message)


pacelogger = PACELogger.pacelogger


def PACE_DEBUG(
    message: str,
    extra_info: Optional[str] = None,
    caller_stack_depth: Optional[int] = 1,
):
    """
    Logs a debug message.

    Args:
        message: The message to be displayed.
    """
    message = f"pace{extra_info if extra_info else ''}: {message}"
    pacelogger(logLevel.DEBUG, message, caller_stack_depth)


def PACE_INFO(
    message: str,
    extra_info: Optional[str] = None,
    caller_stack_depth: Optional[int] = 1,
):
    """
    Logs an informational message.

    Args:
        message: The message to be displayed.
    """

    message = f"pace{extra_info if extra_info else ''}: {message}"
    pacelogger(logLevel.INFO, message, caller_stack_depth)


def PACE_WARNING(
    message: str,
    extra_info: Optional[str] = None,
    caller_stack_depth: Optional[int] = 1,
):
    """
    Logs a warning message.

    Args:
        message: The message to be displayed.
    """

    message = f"pace{extra_info if extra_info else ''}: {message}"
    pacelogger(logLevel.WARNING, message, caller_stack_depth)


def PACE_ASSERT(
    condition: bool,
    message: str,
    extra_info: Optional[str] = None,
    caller_stack_depth: Optional[int] = 1,
):
    """
    Asserts a condition and raises an exception if it is not met.

    Args:
        condition: The condition to be checked.
        message: The message to be displayed if the condition is not met.

    Raises:
        AssertionError: If the condition is not met.
    """

    if not condition:
        err_message = f"pace{extra_info if extra_info else ''}: {message}"
        pacelogger(logLevel.ERROR, err_message, caller_stack_depth)
        raise AssertionError(message)


def PACE_LLM_DEBUG(message: str, caller_stack_depth: Optional[int] = 2):
    PACE_DEBUG(message, extra_info="-llm ", caller_stack_depth=caller_stack_depth)


def PACE_LLM_INFO(message: str, caller_stack_depth: Optional[int] = 2):
    PACE_INFO(message, extra_info="-llm ", caller_stack_depth=caller_stack_depth)


def PACE_LLM_WARNING(message: str, caller_stack_depth: Optional[int] = 2):
    PACE_WARNING(message, extra_info="-llm ", caller_stack_depth=caller_stack_depth)


def PACE_LLM_ASSERT(
    condition: bool, message: str, caller_stack_depth: Optional[int] = 2
):
    PACE_ASSERT(
        condition, message, extra_info="-llm ", caller_stack_depth=caller_stack_depth
    )


@contextmanager
def suppress_logging():
    """
    A context manager that temporarily suppresses all logging.
    This is useful when you want to suppress logging for a specific block of code
    or while testing.
    """
    # Store the original logging level
    original_state = PACELogger.supressor
    try:
        PACELogger.supressor = True
        yield
    finally:
        PACELogger.supressor = original_state


def suppress_logging_fn(func):
    """
    A decorator that suppresses logging for the decorated function.
    """

    def wrapper(*args, **kwargs):
        with suppress_logging():  # Use the context manager
            return func(*args, **kwargs)

    return wrapper


def suppress_logging_cls():
    """
    A decorator that suppresses logging for all methods of the decorated class.
    """

    def wrapper(cls):
        for name, method in cls.__dict__.items():
            if callable(method) and not name.startswith("__"):
                setattr(cls, name, suppress_logging_fn(method))
        return cls

    return wrapper
