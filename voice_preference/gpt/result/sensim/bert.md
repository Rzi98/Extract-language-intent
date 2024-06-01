## Result - BERT (no prompts)

|     | correction                                                                                                                |   tokens | closest_sentence                       |   confidence_score | predicted_feature            | gt_feature                                                                                                      | matches   | cmd_type   |
|----:|:--------------------------------------------------------------------------------------------------------------------------|---------:|:---------------------------------------|-------------------:|:-----------------------------|:----------------------------------------------------------------------------------------------------------------|:----------|:-----------|
|   1 | Get close to bottle and turn right then move closer to apple and stay away from book                                      |       19 | Move further away from bottle          |               0.61 | bottle distance increase     | ['bottle distance decrease', 'X-cartesian increase', 'apple distance decrease', 'book distance increase']       | False     | complex    |
|   2 | Approach the vase and move left then get nearer to the orange and stay down                                               |       17 | Move closer to vase                    |               0.72 | vase distance decrease       | ['vase distance decrease', 'X-cartesian decrease', 'orange distance decrease', 'Z-cartesian decrease']          | False     | complex    |
|   3 | Reduce the distance to the box and turn left then get a bit closer to the banana but avoid the pen                        |       23 | Move closer to banana                  |               0.71 | banana distance decrease     | ['box distance decrease', 'X-cartesian decrease', 'banana distance decrease', 'pen distance increase']          | False     | complex    |
|   4 | Move closer to the cup and go right then keep close to the pear but step back from the notebook                           |       22 | Move closer to pear                    |               0.68 | pear distance decrease       | ['cup distance decrease', 'X-cartesian increase', 'pear distance decrease', 'notebook distance increase']       | False     | complex    |
|   5 | Stay close to the jar and move leftward then decrease the distance to the watermelon but avoid the magazine               |       24 | Keep a bigger distance from watermelon |               0.73 | watermelon distance increase | ['jar distance decrease', 'X-cartesian decrease', 'watermelon distance decrease', 'magazine distance increase'] | False     | complex    |
|   6 | Get nearer to the bottle and approach the monitor then get a little bit lower and stay down                               |       20 | Move closer to bottle                  |               0.68 | bottle distance decrease     | ['bottle distance decrease', 'monitor distance decrease', 'Z-cartesian decrease', 'Z-cartesian decrease']       | False     | complex    |
|   7 | Retreat from the table and move rightwards then move down and keep your distance from the textbook                        |       21 | Move further away from textbook        |               0.71 | textbook distance increase   | ['table distance increase', 'X-cartesian increase', 'Z-cartesian decrease', 'textbook distance increase']       | False     | complex    |
|   8 | Get closer to the plate and go forward then move closer to the peach but steer clear of the journal                       |       22 | Move closer to peach                   |               0.68 | peach distance decrease      | ['plate distance decrease', 'Y-cartesian increase', 'peach distance decrease', 'journal distance increase']     | False     | complex    |
|   9 | Retreat from the glass and stay left then increase the distance from the mango and move a bit further away from the diary |       25 | Move further away from mango           |               0.65 | mango distance increase      | ['glass distance increase', 'X-cartesian decrease', 'mango distance increase', 'diary distance increase']       | False     | complex    |
|  10 | Keep a greater distance from the bowl and move right then move up and avoid the notebook                                  |       19 | Move further away from bowl            |               0.67 | bowl distance increase       | ['bowl distance increase', 'X-cartesian increase', 'Z-cartesian increase', 'notebook distance increase']        | False     | complex    |
|  11 | Traverse right and get closer to the desktop then stay elevated and go backwards                                          |       16 | Move further away from desktop         |               0.74 | desktop distance increase    | ['X-cartesian increase', 'desktop distance decrease', 'Z-cartesian increase', 'Y-cartesian decrease']           | False     | complex    |
|  12 | Move nearer to the book and go rightward then go down and stay close to the book                                          |       20 | Move closer to book                    |               0.79 | book distance decrease       | ['book distance decrease', 'X-cartesian increase', 'Z-cartesian decrease', 'book distance decrease']            | False     | complex    |
|  13 | Get a bit closer to the can and move leftward then descend downward and avoid the magazine                                |       20 | Keep a smaller distance from magazine  |               0.65 | magazine distance decrease   | ['can distance decrease', 'X-cartesian decrease', 'Z-cartesian decrease', 'magazine distance increase']         | False     | complex    |
|  14 | Stay near the cup and move rightwards then move lower and increase the distance from the notebook                         |       21 | Keep a bigger distance from cup        |               0.66 | cup distance increase        | ['cup distance decrease', 'X-cartesian increase', 'Z-cartesian decrease', 'notebook distance increase']         | False     | complex    |
|  15 | Get closer to the plate and approach the laptop then remain at a lower position and keep your distance from the journal   |       24 | Stay close to notebook                 |               0.62 | notebook distance decrease   | ['plate distance decrease', 'laptop distance decrease', 'Z-cartesian decrease', 'journal distance increase']    | False     | complex    |
|  16 | Move away from the glass and stay left then go up and move a bit further away from the diary                              |       22 | Move closer to diary                   |               0.7  | diary distance decrease      | ['glass distance increase', 'X-cartesian decrease', 'Z-cartesian increase', 'diary distance increase']          | False     | complex    |
|  17 | Keep a greater distance from the bowl and move rightwards then ascend upwards and avoid the notebook                      |       22 | Move further away from bowl            |               0.66 | bowl distance increase       | ['bowl distance increase', 'X-cartesian increase', 'Z-cartesian increase', 'notebook distance increase']        | False     | complex    |
|  18 | Move nearer to the jar and go rightward then stay elevated and avoid the textbook                                         |       18 | Move further away from textbook        |               0.65 | textbook distance increase   | ['jar distance decrease', 'X-cartesian increase', 'Z-cartesian increase', 'textbook distance increase']         | False     | complex    |
|  19 | Move closer to the plate and get close to the desktop then move lower and steer clear of the magazine                     |       22 | Move closer to magazine                |               0.67 | magazine distance decrease   | ['plate distance decrease', 'desktop distance decrease', 'Z-cartesian decrease', 'magazine distance increase']  | False     | complex    |
|  20 | Move forward and move rightwards then decrease the distance to the notebook and move a bit further away from the diary    |       25 | Move closer to diary                   |               0.76 | diary distance decrease      | ['Y-cartesian increase', 'X-cartesian increase', 'notebook distance decrease', 'diary distance increase']       | False     | complex    |
|  21 | move more left and stay closer to the bottle                                                                              |       11 | Move closer to bottle                  |               0.89 | bottle distance decrease     | ['X-cartesian decrease', 'bottle distance decrease']                                                            | False     | complex    |
|  22 | stay closer to the laptop and cup, move up a little bit                                                                   |       15 | Move closer to laptop                  |               0.66 | laptop distance decrease     | ['laptop distance decrease', 'cup distance decrease', 'Z-cartesian increase']                                   | False     | complex    |
|  23 | move down and closer to the plate and microwave                                                                           |       11 | Move closer to microwave               |               0.79 | microwave distance decrease  | ['Z-cartesian decrease', 'Plate distance decrease', 'Microwave distance decrease']                              | False     | complex    |
|  24 | avoid the cake and the fridge                                                                                             |        8 | Avoid cake                             |               0.79 | cake distance increase       | ['cake distance increase', 'fridge distance increase']                                                          | False     | complex    |
|  25 | move away from bottle and book then move closer to cup                                                                    |       13 | Move closer to bottle                  |               0.79 | bottle distance decrease     | ['bottle distance increase', 'book distance increase', 'cup distance decrease']                                 | False     | complex    |
|  26 | move away from bottle and then move closer to cup                                                                         |       12 | Move closer to bottle                  |               0.84 | bottle distance decrease     | ['bottle distance increase', 'cup distance decrease']                                                           | False     | complex    |
|  27 | move closer to egg and cake                                                                                               |        8 | Move closer to cake                    |               0.86 | cake distance decrease       | ['Egg distance decrease', 'Cake distance decrease']                                                             | False     | complex    |
|  28 | move closer to the bottle and move further from the laptop                                                                |       13 | Move further away from bottle          |               0.77 | bottle distance increase     | ['bottle distance decrease', 'laptop distance increase']                                                        | False     | complex    |
|  29 | move up                                                                                                                   |        4 | Move up                                |               1    | Z-cartesian increase         | ['Z-cartesian increase']                                                                                        | True      | single     |
|  30 | move right                                                                                                                |        4 | Move right                             |               1    | X-cartesian increase         | ['X-cartesian increase']                                                                                        | True      | single     |
|  31 | keep close to pineapple                                                                                                   |        7 | Stay close to pineapple                |               0.98 | pineapple distance decrease  | ['Pineapple distance decrease']                                                                                 | True      | single     |
|  32 | move near the bottle                                                                                                      |        6 | Move closer to bottle                  |               0.91 | bottle distance decrease     | ['Bottle distance decrease']                                                                                    | True      | single     |
|  33 | move near the pineapple                                                                                                   |        7 | Move closer to pineapple               |               0.92 | pineapple distance decrease  | ['Pineapple distance decrease']                                                                                 | True      | single     |
|  34 | move further from the cake                                                                                                |        7 | Move further away from cake            |               0.95 | cake distance increase       | ['Cake distance increase']                                                                                      | True      | single     |
|  35 | move closer to the fridge                                                                                                 |        7 | Move closer to fridge                  |               0.98 | fridge distance decrease     | ['Fridge distance decrease']                                                                                    | True      | single     |
|  36 | keep right and stay closer to the bottle                                                                                  |       10 | Stay close to bottle                   |               0.88 | bottle distance decrease     | ['X-cartesian increase', 'Bottle distance decrease']                                                            | False     | complex    |
|  37 | stay closer to the vase and cup, move up a little bit                                                                     |       15 | Move closer to vase                    |               0.75 | vase distance decrease       | ['Vase distance decrease', 'Cup distance decrease', 'Z-cartesian increase']                                     | False     | complex    |
|  38 | move lower and closer to the banana and apple                                                                             |       11 | Move closer to banana                  |               0.85 | banana distance decrease     | ['Z-cartesian decrease', 'Banana distance decrease', 'Apple distance decrease']                                 | False     | complex    |
|  39 | stay extremely far away from the bottle and cup                                                                           |       11 | Stay close to bottle                   |               0.81 | bottle distance decrease     | ['Bottle distance increase', 'Cup distance increase']                                                           | False     | complex    |
|  40 | stay away from laptop                                                                                                     |        6 | Stay away from laptop                  |               1    | laptop distance increase     | ['Laptop distance increase']                                                                                    | True      | single     |
|  41 | stay further away from the pineapple                                                                                      |        9 | Stay close to pineapple                |               0.93 | pineapple distance decrease  | ['Pineapple distance increase']                                                                                 | False     | single     |
|  42 | stay closer to spoon                                                                                                      |        6 | Stay close to spoon                    |               0.98 | spoon distance decrease      | ['Spoon distance decrease']                                                                                     | True      | single     |
|  43 | move upwards                                                                                                              |        4 | Move higher                            |               0.89 | Z-cartesian increase         | ['Z-cartesian increase']                                                                                        | True      | single     |
|  44 | move further from laptop                                                                                                  |        6 | Move further away from laptop          |               0.95 | laptop distance increase     | ['Laptop distance increase']                                                                                    | True      | single     |
|  45 | move closer to pineapple                                                                                                  |        7 | Move closer to pineapple               |               1    | pineapple distance decrease  | ['Pineapple distance decrease']                                                                                 | True      | single     |
|  46 | move nearer to spoon                                                                                                      |        6 | Move closer to spoon                   |               0.98 | spoon distance decrease      | ['Spoon distance decrease']                                                                                     | True      | single     |
|  47 | move lower                                                                                                                |        4 | Move lower                             |               1    | Z-cartesian decrease         | ['Z-cartesian decrease']                                                                                        | True      | single     |
|  48 | avoid cake                                                                                                                |        4 | Avoid cake                             |               1    | cake distance increase       | ['Cake distance increase']                                                                                      | True      | single     |
|  49 | move away from bottle and then closer to cup                                                                              |       11 | Move closer to bottle                  |               0.84 | bottle distance decrease     | ['Bottle distance increase', 'Cup distance decrease']                                                           | False     | complex    |
|  50 | move closer to spoon                                                                                                      |        6 | Move closer to spoon                   |               1    | spoon distance decrease      | ['Spoon distance decrease']                                                                                     | True      | single     |
|  51 | move closer to microwave                                                                                                  |        6 | Move closer to microwave               |               1    | microwave distance decrease  | ['Microwave distance decrease']                                                                                 | True      | single     |
|  52 | retreat from the bottle                                                                                                   |        6 | Move further away from bottle          |               0.73 | bottle distance increase     | ['Bottle distance increase']                                                                                    | True      | single     |
|  53 | move towards bottle                                                                                                       |        5 | Move closer to bottle                  |               0.9  | bottle distance decrease     | ['Bottle distance decrease']                                                                                    | True      | single     |
|  54 | move away from laptop                                                                                                     |        6 | Move further away from laptop          |               0.9  | laptop distance increase     | ['Laptop distance increase']                                                                                    | True      | single     |
|  55 | move away from pineapple                                                                                                  |        7 | Move further away from pineapple       |               0.94 | pineapple distance increase  | ['Pineapple distance increase']                                                                                 | True      | single     |
|  56 | move towards spoon                                                                                                        |        5 | Move closer to spoon                   |               0.9  | spoon distance decrease      | ['Spoon distance decrease']                                                                                     | True      | single     |
|  57 | move towards cake                                                                                                         |        5 | Move further away from cake            |               0.88 | cake distance increase       | ['Cake distance decrease']                                                                                      | False     | single     |
|  58 | move towards egg                                                                                                          |        5 | Move closer to egg                     |               0.89 | egg distance decrease        | ['Egg distance decrease']                                                                                       | True      | single     |
|  59 | go close to cup                                                                                                           |        6 | Stay close to cup                      |               0.9  | cup distance decrease        | ['Cup distance decrease']                                                                                       | True      | single     |
|  60 | move down                                                                                                                 |        4 | Move down                              |               1    | Z-cartesian decrease         | ['Z-cartesian decrease']                                                                                        | True      | single     |
|  61 | move to pineapple                                                                                                         |        6 | Move further away from pineapple       |               0.86 | pineapple distance increase  | ['Pineapple distance decrease']                                                                                 | False     | single     |
|  62 | move to pineapple, move down                                                                                              |        9 | Move further away from pineapple       |               0.85 | pineapple distance increase  | ['Pineapple distance decrease', 'Z-cartesian decrease']                                                         | False     | complex    |
|  63 | avoid the fridge                                                                                                          |        5 | Avoid fridge                           |               0.97 | fridge distance increase     | ['Fridge distance increase']                                                                                    | True      | single     |
|  64 | closer to cake                                                                                                            |        5 | Stay close to cake                     |               0.82 | cake distance decrease       | ['Cake distance decrease']                                                                                      | True      | single     |
|  65 | move closer to bottle                                                                                                     |        6 | Move closer to bottle                  |               1    | bottle distance decrease     | ['Bottle distance decrease']                                                                                    | True      | single     |
|  66 | avoid the cake                                                                                                            |        5 | Avoid cake                             |               0.96 | cake distance increase       | ['Cake distance increase']                                                                                      | True      | single     |
|  67 | move closer to egg                                                                                                        |        6 | Move closer to egg                     |               1    | egg distance decrease        | ['Egg distance decrease']                                                                                       | True      | single     |
|  68 | move towards the bottle                                                                                                   |        6 | Move closer to bottle                  |               0.88 | bottle distance decrease     | ['Bottle distance decrease']                                                                                    | True      | single     |
|  69 | move closer to the spoon                                                                                                  |        7 | Move closer to spoon                   |               0.98 | spoon distance decrease      | ['Spoon distance decrease']                                                                                     | True      | single     |
|  70 | move away from the microwave                                                                                              |        7 | Move further away from microwave       |               0.91 | microwave distance increase  | ['Microwave distance increase']                                                                                 | True      | single     |
|  71 | move towards the egg                                                                                                      |        6 | Move closer to egg                     |               0.86 | egg distance decrease        | ['Egg distance decrease']                                                                                       | True      | single     |
|  72 | left                                                                                                                      |        3 | Left                                   |               1    | X-cartesian decrease         | ['X-cartesian decrease']                                                                                        | True      | single     |
|  73 | up                                                                                                                        |        3 | Up                                     |               1    | Z-cartesian increase         | ['Z-cartesian increase']                                                                                        | True      | single     |
|  74 | move to the cake                                                                                                          |        6 | Move further away from cake            |               0.83 | cake distance increase       | ['Cake distance decrease']                                                                                      | False     | single     |
|  75 | avoid the egg                                                                                                             |        5 | Avoid egg                              |               0.96 | egg distance increase        | ['Egg distance increase']                                                                                       | True      | single     |
|  76 | move closer to the cup                                                                                                    |        7 | Move closer to cup                     |               0.94 | cup distance decrease        | ['Cup distance decrease']                                                                                       | True      | single     |
|  77 | move closer to the pineapple                                                                                              |        8 | Move closer to pineapple               |               0.98 | pineapple distance decrease  | ['Pineapple distance decrease']                                                                                 | True      | single     |
|  78 | move closer to the egg                                                                                                    |        7 | Move closer to egg                     |               0.97 | egg distance decrease        | ['Egg distance decrease']                                                                                       | True      | single     |
|  79 | move higher                                                                                                               |        4 | Move higher                            |               1    | Z-cartesian increase         | ['Z-cartesian increase']                                                                                        | True      | single     |
|  80 | move closer to the bottle                                                                                                 |        7 | Move closer to bottle                  |               0.96 | bottle distance decrease     | ['Bottle distance decrease']                                                                                    | True      | single     |
|  81 | move away from the pineapple                                                                                              |        8 | Move further away from pineapple       |               0.93 | pineapple distance increase  | ['Pineapple distance increase']                                                                                 | True      | single     |
|  82 | move away from the cake                                                                                                   |        7 | Move further away from cake            |               0.92 | cake distance increase       | ['Cake distance increase']                                                                                      | True      | single     |
|  83 | closer to the bottle and further from the laptop                                                                          |       11 | Move closer to bottle                  |               0.69 | bottle distance decrease     | ['Bottle distance decrease', 'Laptop distance increase']                                                        | False     | complex    |
|  84 | keep close to the table but stay clear from cup                                                                           |       12 | Stay close to cup                      |               0.78 | cup distance decrease        | ['Table distance decrease', 'Cup distance increase']                                                            | False     | complex    |
|  85 | closer to the cake                                                                                                        |        6 | Stay close to cake                     |               0.85 | cake distance decrease       | ['Cake distance decrease']                                                                                      | True      | single     |
|  86 | move away from the laptop but closer to the cup                                                                           |       12 | Move closer to laptop                  |               0.71 | laptop distance decrease     | ['Laptop distance increase', 'Cup distance decrease']                                                           | False     | complex    |
|  87 | move to the right                                                                                                         |        6 | Move left                              |               0.92 | X-cartesian decrease         | ['X-cartesian increase']                                                                                        | False     | single     |
|  88 | move towards the spoon                                                                                                    |        6 | Move closer to spoon                   |               0.89 | spoon distance decrease      | ['Spoon distance decrease']                                                                                     | True      | single     |
|  89 | Get closer to the can                                                                                                     |        7 | Keep a bigger distance from can        |               0.86 | can distance increase        | ['Can distance decrease']                                                                                       | False     | single     |
|  90 | Traverse right                                                                                                            |        4 | Keep a bigger distance from can        |               0.86 | can distance increase        | ['X-cartesian increase']                                                                                        | False     | single     |
|  91 | Move forward and then move right                                                                                          |        8 | Move backward                          |               0.79 | Y-cartesian decrease         | ['Y-cartesian increase', 'X-cartesian increase']                                                                | False     | complex    |
|  92 | Move backward and then move up                                                                                            |        8 | Move backward                          |               0.86 | Y-cartesian decrease         | ['Y-cartesian decrease', 'Z-cartesian increase']                                                                | False     | complex    |
|  93 | Approach the cup but avoid the scissors and keep left                                                                     |       12 | Avoid scissors                         |               0.68 | scissors distance increase   | ['Cup distance decrease', 'Scissors distance increase', 'X-cartesian decrease']                                 | False     | complex    |
|  94 | Go close to the banana but keep a distance from apple                                                                     |       13 | Stay close to banana                   |               0.87 | banana distance decrease     | ['Banana distance decrease', 'Apple distance increase']                                                         | False     | complex    |
|  95 | Move towards the cup and keep left                                                                                        |        9 | Move closer to cup                     |               0.8  | cup distance decrease        | ['Cup distance decrease', 'X-cartesian decrease']                                                               | False     | complex    |
|  96 | Move a little bit forward                                                                                                 |        7 | Move forward                           |               0.8  | Y-cartesian increase         | ['Y-cartesian increase']                                                                                        | True      | single     |
|  97 | Stay nearer to cup but keep a distance from the scissors                                                                  |       13 | Stay close to scissors                 |               0.81 | scissors distance decrease   | ['Cup distance decrease', 'Scissors distance increase']                                                         | False     | complex    |
|  98 | Stay far from the book and keep right while moving forward                                                                |       13 | Move further away from book            |               0.81 | book distance increase       | ['Book distance increase', 'X-cartesian increase', 'Y-cartesian increase']                                      | False     | complex    |
|  99 | Move towards the book and keep back from the banana                                                                       |       12 | Move further away from banana          |               0.71 | banana distance increase     | ['book distance decrease', 'banana distance increase']                                                          | False     | complex    |
| 100 | Take a step forward                                                                                                       |        6 | Move forward                           |               0.82 | Y-cartesian increase         | ['Y-cartesian increase']                                                                                        | True      | single     |

## Stats

| matches   |   single_cmd |   chain_cmd |
|:----------|-------------:|------------:|
| False     |            7 |          45 |
| True      |           48 |           0 |