"""
A tiny utility lib to handle functions raising Exceptions
"""

import typing


def on_fail_ask_user_politely(f: typing.Callable):
    """
    A UEXP wrapper that makes it so an app won't collapse by a single failed function call
    :param f: the function that can't panic
    :return: the wrapped f
    """
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print("\n\033[31;1m Execution failed.\033[0m\n\tDo you want to know why ?\n\t")
            if "y" in input().lower():
                raise e
            else:
                return e
        except KeyboardInterrupt:
            print("\n\033[31;1mInterrupted\033[0m with success\n")
    return wrapper
