from typing import List, Iterable, Tuple, Union

def normalized_var_to_hide_and_model_vars(var_to_hide: Union[int, Iterable[int]], F: int) -> Tuple[List[int], List[int]]:
    F = int(F)
    vars_to_hide = []

    iterable_passed = isinstance(var_to_hide, Iterable)

    for idx, v in enumerate(var_to_hide if iterable_passed else [var_to_hide]):
        v = int(v)
        if not -F <= v < F:
            error_msg = "var_to_hide"
            if iterable_passed:
                error_msg += f"[{idx}]"
            error_msg += f"={var_to_hide} is not in permissible range by F={F}, namely [{-F}, {F-1}]"
            raise IndexError(error_msg)
        
        vars_to_hide.append(v % F)
    
    model_vars = [f for f in range(F) if f!=vars_to_hide]

    return vars_to_hide, model_vars