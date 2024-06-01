temp = {
    "0": {
        "id": 1,
        "correction": "Get close to bottle and turn right then move closer to apple and stay away from book",
        "obj_names": ["apple", "bottle", "book", "minibus", "oranges"],# 5 random objects with 1 correct object
        "obj_classes": [1,2,3],    # latte's label
        "change_type": "dist",  # dist/spd  # latte's label    
        "similarity": [[0.5, 0.2, 0.3]],
        "gt_target_object": ["obj1"],
        "gt_direction": ["decrease"],
        "gt_intensity": ["neutral"],   #"high", "neutral", "low"
        "gt_cart_axes": ["-"],
        "gt_change_type": ["distance"],
        "gt_feature": ["obj1_distance_decrease"],
        "gt_dynamic_features": ["ft1", "ft2"],   
        "gt_split": ["correction"]# split cmds
    }
}

