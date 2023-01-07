def boxed_string(str_):
    line = '-' * len(str_)
    msg = f'\n  +--{line}--+' \
          f'\n  |  {str_}  |' \
          f'\n  +--{line}--+'
    return msg
