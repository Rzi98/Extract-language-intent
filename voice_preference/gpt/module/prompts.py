dynamic_feature_prompt = """
Append physical/tangible object found in the input into a list. Return empty list if no object found. 
Separate each object with a comma.

Ensure object is in singular form. (e.g. "bottles" -> "bottle")
EXCEPTION (scissors -> scissors | pants -> pants | glasses -> glasses)

FORMAT JSON:
    {
        "obj_list": [__, ...]
    }
"""


split_prompt = """
Analyze the user's command. If the command contains multiple instructions, break it down into simpler sentences and separate them with a comma without any newline. 
Otherwise return the same input. Separate each sentence with a comma without any newline. 

ADDITIONALLY:
check if the input is DISTANCE or SPEED related.

FORMAT JSON:
    {
        "split": [__, ...],
        "type" : ["DISTANCE" or "SPEED", ...]
    }
"""

inference_prompt = """
Return the closest semantic feature from Features that matches each command in a list. 
If approach an object, object distance should decrease. 
If avoid an object, object distance should increase. 
Output it into a list separated by comma. Length of output == length of input.
Output must be from Features.

REFERENCE AXES (when commands does not involve object):
Key: left , Feature: X-cartesian decrease 
Key: right , Feature: X-cartesian increase
Key: up , Feature: Z-cartesian increase
Key: down , Feature: Z-cartesian decrease
Key: forward , Feature: Y-cartesian increase
Key: backward , Feature: Y-cartesian decrease


ADDITIONALLY:
- return the confidence score for each prediction.

FORMAT JSON:
        {
            "output" : [__, __, __, ...],
            "confidence": [__, __, __, ...]
        }
"""

enhanced_inference_prompt = """
Return the closest semantic feature from the provided DYNAMIC FEATURES list that matches each command in a given list. 
If a command indicates approaching an object, the object's distance decreases. 
If a command implies avoiding an object or turning away from it, the object's distance increases. 
If a command involves changes in speed, the appropriate 'speed increase' or 'speed decrease' feature should be issued. 
Commands that include directional movement WITHOUT a specified object should be matched with the corresponding Cartesian axis feature. 
Compound commands require outputting multiple features.

The output array must have a length equal to the number of commands in the input. Compound commands should have their respective features concatenated in the output array, separated by a comma within ONE STRING.
EXAMPLE: (speed after distance)
input: "Go fast when close to bottle"
out: "bottle distance decrease, speed increase"

For compound commands, assess and include both the speed alteration and (cartesian axis change or object distance change).

REFERENCE AXES (when commands does not involve object):
- Key: left , Feature: X-cartesian decrease
- Key: right , Feature: X-cartesian increase
- Key: up , Feature: Z-cartesian increase
- Key: down , Feature: Z-cartesian decrease
- Key: forward , Feature: Y-cartesian increase
- Key: backward , Feature: Y-cartesian decrease

ADDITIONALLY:
- return the confidence score for each prediction.

JSON output should follow this format:
{
    "output" : ["___", ...],
    "confidence": [___, ...]
}
"""

intensity_prompt = """
Categorize each command given into one of the following intensities. Ensure each command is matched with an intensity.

1. HIGH  - when the input has high intensity description on the instructions like "very", "a lot", "greater" or anything that signifies MORE.

2. LOW  - is the opposite, low intensity description on instructions like "a little bit", "not by much", "lesser", something that signifies LESS

3. NEUTRAL - is when the input does not have any of the above description in its input.

Return only the answer.

FORMAT JSON:
    {
        "intensity" : [__, __, __, ...],
    }
"""


enhanced_example_selector = """
DYNAMIC FEATURES: ['bottle distance increase', 'bottle distance decrease', 'apple distance increase', 'apple distance decrease', 'book distance increase', 'book distance decrease', 'speed increase', 'speed decrease']

For example, given the commands: ['Get close to bottle', 'Move slower when turning right', 'Move closer to apple', 'Go faster around the book'], your output should be:
{
    "output": ["bottle distance decrease", "speed decrease, X-cartesian increase", "apple distance decrease", "speed increase"],
    "confidence": [0.95, 0.75, 0.95, 0.9]
}
"""