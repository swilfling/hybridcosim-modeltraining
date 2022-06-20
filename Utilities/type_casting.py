def cast(variable, totype: str):
    """
    Cast a variable to a type, where type is a string variable
    Can cast to base types: float, int, bool and string
    Parameters
    ----------
    variable The variable to cast
    totype The type to cast into, as a string:
        Real   -> float
        Int    -> int
        Bool   -> bool
        String -> str

    Returns totype(variable) or original type if type not found
    -------

    """
    castingtype = totype.lower()
    if castingtype == "real":
        return float(variable)
    if castingtype == "int":
        return int(variable)
    if castingtype == "bool":
        return bool(variable)
    if castingtype == "string":
        return str(variable)
    return variable