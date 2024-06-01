OBJECT_PROMPT = """
You will be given an input string and a list of object names.  Compare EACH object in the list to the string.

1. Check if the string even has any object. If NONE found, just return "None" and SKIP step 2 and 
2. SHOW similarity scores for EACH of the object in object names. 
3. Extract out the object with highest similarity score to the string.


Only return the answer, no extra comments.

FORMAT JSON:

{
    similarity {
                   obj: score,
                    ....},
   Closest_obj:  ...
}

OR

{
    similarity {
                   obj: score,
                    ....},
   closest_obj:  "None"
}
"""

INTENSITY_TYPE_PROMPT = """
Categorize the input given into one of the following intensities.

1. high  - when the input has high intensity description on the instructions like "very, a lot" and anything that signifies more.

2. low  - is the opposite, low intensity description on instructions like "a little bit, not by much", something that signifies less

3. neutral - is when the input does not have any of the above description in its input.

ADDITIONALLY:
check if the input is DISTANCE or SPEED RELATED.

Return only the answer.

FORMAT JSON:
{
   intensity : ___,
   type : "DISTANTCE" or "SPEED"
}
"""

CLASSIFIER_PROMPT = """
You will be given an input string and a list of features.
Select one feature that has the best match to the input.
"""

CARTESIAN_REFERENCE_TEMPLATE = """
Cartesian Reference:
leftward - X-cartesian decrease
rightward - X-cartesian decrease
backward - Y-cartesian decrease
forward - "-cartesian decrease
downwards - Z-cartesian decrease
upwards - Z-cartesian decrease
"""

CLASSIFIER_FORMAT = """
Return FORMAT:
{
    "output" : ____
}
"""

if __name__ == '__main__':
   print(INTENSITY_TYPE_PROMPT)