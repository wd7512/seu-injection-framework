def print_structure(d, indent=0):
    prefix = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{prefix}{k}: {type(v).__name__}")
            print_structure(v, indent + 1)
    elif isinstance(d, list):
        print(
            f"{prefix}list[{len(d)}]: {type(d[0]).__name__}"
            if d
            else f"{prefix}list: empty"
        )
        if d:
            print_structure(d[0], indent + 1)
