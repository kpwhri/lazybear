import warnings
from functools import wraps

# alert users to any deprecation as they need to alter behavior
warnings.simplefilter('always', DeprecationWarning)


def deprecated_param(param_name, message):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if param_name in kwargs and kwargs[param_name] is not None:
                warnings.warn(
                    message,
                    DeprecationWarning,
                    stacklevel=2
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
