<h2> Table 1: NER </h2>

|    |   id | predicted                                          | truth                                                                                                      |
|---:|-----:|:---------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|
|  1 |   34 | NA                                                 | ['cake distance decrease', 'cake distance increase']                                                       |
|  2 |   44 | NA                                                 | ['laptop distance decrease', 'laptop distance increase']                                                   |
|  3 |   48 | NA                                                 | ['cake distance decrease', 'cake distance increase']                                                       |
|  4 |   55 | NA                                                 | ['pineapple distance decrease', 'pineapple distance increase']                                             |
|  5 |   64 | NA                                                 | ['cake distance decrease', 'cake distance increase']                                                       |
|  6 |   66 | NA                                                 | ['cake distance decrease', 'cake distance increase']                                                       |
|  7 |   86 | ['cup distance decrease', 'cup distance increase'] | ['cup distance decrease', 'cup distance increase', 'laptop distance decrease', 'laptop distance increase'] |

|             |   Score (%) |
|:------------|------------:|
| error_count |           7 |
| total_data  |         100 |
| accuracy    |          93 |<h2> Table 1: NER </h2>

<h2> Table 2: Split </h2>

|    |   id | predicted                                                                                                                                     | truth                                                                                  |
|---:|-----:|:----------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
|  1 |   13 | ['get a bit closer to the can', 'move leftward', 'descend downward', 'and avoid the magazine']                                                | ['get a bit closer to can', 'move leftward', 'descend downward', 'avoid the magazine'] |
|  2 |   27 | ['move closer to the egg', 'move closer to the cake']                                                                                         | ['move closer to egg', 'move closer to cake']                                          |
|  3 |   44 | ['move further away from the laptop']                                                                                                         | ['move further from laptop']                                                           |
|  4 |   48 | ['avoiding cake is a good idea']                                                                                                              | ['avoid cake']                                                                         |
|  5 |   49 | ['move away from bottle', 'then move closer to cup']                                                                                          | ['move away from bottle', 'move closer to cup']                                        |
|  6 |   51 | ['move closer to the microwave']                                                                                                              | ['move closer to microwave']                                                           |
|  7 |   59 | ['go close to the cup']                                                                                                                       | ['go close to cup']                                                                    |
|  8 |   63 | ['avoiding the fridge can be a good idea to prevent unnecessary snacking. it can help in maintaining a healthy diet and avoiding overeating'] | ['avoid the fridge']                                                                   |
|  9 |   75 | ['avoiding the egg is important']                                                                                                             | ['avoid the egg']                                                                      |
| 10 |   94 | ['go close to the banana', 'but keep a distance from the apple']                                                                              | ['go close to the banana', 'keep a distance from apple']                               |
| 11 |   98 | ['stay far from the book', 'keep right', 'move forward']                                                                                      | ['stay far from the book', 'keep right', 'moving forward']                             |

|             |   Score (%) |
|:------------|------------:|
| error_count |          11 |
| total_data  |         100 |
| accuracy    |          89 |<h2> Table 2: Split </h2>

<h2> Table 3: Classification </h2>

|    |   id | predicted                                             | truth                                                 |
|---:|-----:|:------------------------------------------------------|:------------------------------------------------------|
|  1 |   34 | ['z-cartesian decrease']                              | ['cake distance increase']                            |
|  2 |   44 | ['z-cartesian decrease']                              | ['laptop distance increase']                          |
|  3 |   48 | ['z-cartesian increase']                              | ['cake distance increase']                            |
|  4 |   49 | ['bottle distance decrease', 'cup distance increase'] | ['bottle distance increase', 'cup distance decrease'] |
|  5 |   55 | ['z-cartesian decrease']                              | ['pineapple distance increase']                       |
|  6 |   64 | ['z-cartesian increase']                              | ['cake distance decrease']                            |
|  7 |   66 | ['z-cartesian increase']                              | ['cake distance increase']                            |
|  8 |   86 | ['cup distance decrease', 'cup distance increase']    | ['cup distance decrease', 'laptop distance increase'] |

|             |   Score (%) |
|:------------|------------:|
| error_count |           8 |
| total_data  |         100 |
| accuracy    |          92 |<h2> Table 3: Classification </h2>

