import typing


def on_fail_ask_user_politely(f: typing.Callable):
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            print("\033[31;1m Execution failed.\033[0m Do you want to know why ?")
            if "y" in input().lower():
                raise e
            else:
                return
        except KeyboardInterrupt:
            print("\033[31;1mInterrupted\033[0m with success")

    return wrapper
