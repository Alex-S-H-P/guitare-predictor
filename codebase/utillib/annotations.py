import typing


def on_fail_ask_user_politely(f: typing.Callable):
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
