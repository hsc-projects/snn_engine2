def boxed_string(str_, inbox_h_margin=2):
    space = ' ' * inbox_h_margin
    line = '-' * inbox_h_margin
    line = line + '-' * len(str_) + line
    msg = f'\n  +{line}+' \
          f'\n  |{space + str_ + space}|' \
          f'\n  +{line}+'
    return msg
