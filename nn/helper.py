def convert_to_time(value):
    hours = value // 100
    minutes = value % 100
    return f"{hours:02d}:{minutes:02d}"
