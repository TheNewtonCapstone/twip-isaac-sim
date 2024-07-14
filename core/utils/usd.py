import re
from typing import Any

def find_matching_prims(prim_path_regex: str, stage: Any | None = None) -> list[Any]:
    """Find all the matching prims in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list of prims that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    from omni.isaac.core.utils.stage import get_current_stage
    
    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")
    
    # get current stage
    if stage is None:
        stage = get_current_stage()
    
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    tokens = prim_path_regex.split("/")[1:]
    tokens = [f"^{token}$" for token in tokens]
    
    # iterate over all prims in stage (breath-first search)
    all_prims = [stage.GetPseudoRoot()]
    output_prims = []
    for index, token in enumerate(tokens):
        token_compiled = re.compile(token)
        for prim in all_prims:
            for child in prim.GetAllChildren():
                if token_compiled.match(child.GetName()) is not None:
                    output_prims.append(child)
        if index < len(tokens) - 1:
            all_prims = output_prims
            output_prims = []
    
    return output_prims

def find_first_matching_prim(prim_path_regex: str, stage: Any | None = None) -> Any | None:
    """Find the first matching prim in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first prim that matches input expression. If no prim matches, returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    from omni.isaac.core.utils.stage import get_current_stage
    
    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")
 
    # get current stage
    if stage is None:
        stage = get_current_stage()
 
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    pattern = f"^{prim_path_regex}$"
    compiled_pattern = re.compile(pattern)
 
    # obtain matching prim (depth-first search)
    for prim in stage.Traverse():
        # check if prim passes predicate
        if compiled_pattern.match(prim.GetPath().pathString) is not None:
            return prim

    return None

def find_first_matching_prim(prim_path_regex: str, stage: Any | None = None) -> Any | None:
    """Find the first matching prim in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first prim that matches input expression. If no prim matches, returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    from omni.isaac.core.utils.stage import get_current_stage
    
    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")
 
    # get current stage
    if stage is None:
        stage = get_current_stage()
 
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    pattern = f"^{prim_path_regex}$"
    compiled_pattern = re.compile(pattern)
 
    # obtain matching prim (depth-first search)
    for prim in stage.Traverse():
        # check if prim passes predicate
        if compiled_pattern.match(prim.GetPath().pathString) is not None:
            return prim
 
    return None