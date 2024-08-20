HIFIBER_MAPPINGS = {
    'fibertree.Tensor': {
        "path": "hifiber::core::tensor::Tensor",
        "methods": {
            "swizzleRanks": {
                "name": "swizzle_ranks",
                "return_type": "hifiber::core::tensor::Tensor",
            },
            "getRoot": {
                "name": "get_root_mut",
                "return_type": "&mut hifiber::core::eager::EagerFiber",
                "hide_annotation": False,
            },
            "setRankIds": {
                "name": "set_rank_ids",
                "return_type": None,
            }
        }
    },
    
    'fibertree.Fiber': {
        "path": "hifiber::core::eager::EagerFiber",
        "methods": {}
    },
    'fibertree.Metrics': None,
    'teaal.parse.Einsum': None,
    'teaal.parse.Mapping': None,
    'teaal.parse.Architecture': None,
    'teaal.parse.Bindings': None,
    'teaal.parse.Format': None,
}

def contract_type_name(type_name) -> str:
    type_name_split = type_name.split("::")
    ref_type = type_name_split[0].split(" ")[0]
    object_type = type_name_split[-1]

    if ref_type.startswith('&'):
        return ref_type +' ' + object_type
    else:
        return object_type

def extract_type_object(type_name) -> str:
    return type_name.split(" ")[-1]

def map_import(class_path):
    if class_path in HIFIBER_MAPPINGS:
        rs_class = HIFIBER_MAPPINGS[class_path]
        rs_class_path = rs_class["path"]
        return rs_class_path
    else:
        return None

def py_method_name_to_rs_method(py_method_name):
    for py_class_path in HIFIBER_MAPPINGS:
        py_class = HIFIBER_MAPPINGS[py_class_path]
        if py_class == None:
            continue

        py_method_names = py_class["methods"]
        if py_method_name in py_method_names:
            return py_class["methods"][py_method_name]
    
    return None

def py_class_name_to_rs_class(py_class_name):
    for py_class_path in HIFIBER_MAPPINGS:
        _py_class_name = py_class_path.split(".")[-1]
        if _py_class_name == py_class_name:
            return HIFIBER_MAPPINGS[py_class_path]
            #rs_class_path = HIFIBER_CLASSES[py_class_path]["path"]
            #rs_class_name = rs_class_path.split("::")[-1]
            #return rs_class_name

    return None
