基于sklearn的文本分类器 Text classifier based on sklearn
=======================================================
Short text Classifier based on Numpy,scikit-learn,Pandas,Matplotlib

Train Data Format
----------------------
|   **type**  |                     **Text**                        |
|:-----------:|:---------------------------------------------------:|
|     game    |   The LoL champions pro players would ban forever   |
|     society |   In Beijing you should keep the rules              |
|     etc.    |   etc.                                              |

Sample Usage
----------------------
```python
>>> import TextClassifier

    # cerat classifier container
>>> tc = TextClassifier.classifier_container()

    # load data.sep Default = ',' you can change it to '\t',etc.
>>> tc.load_Data('../data/Train_data.txt',sep=',')

    # train the model
>>> tc.train()

    # prediction. Input list or text-String
>>> print tc.predict('Faker is the first League of Legends player to earn over $1 million in prize money')
>>> [u'game']
>>> print tc.predict(['Faker is the first League of Legends player to earn over $1 million in prize money',
                    '18-year-old youth killed 88-year-old veteran',
                    'Take you into the real North Korea'])
>>> [u'game',u'society',u'world']

    #get X_train, X_test, y_train, y_test
>>> from sklearn import cross_validation
>>> X_train, X_test, y_train, y_test = cross_validation.train_test_split(original_data['Text'], original_data['Categorization'], test_size=0.3, random_state=0)

    #get TrainData Accuracy
>>> tc.Accuracy(X_train, y_train)
>>> Accuracy:
    0.917504310503

    #get Confusion Matrix
>>> Y_predict = tc.predict(X_test)
>>> tc.confusion_matrix(y_test, Y_predict)

```
```python
>>> Confusion Matrix :
               military  baby   car  game  food  sports  finance  discovery  regimen  travel  fashion  history  society  story  tech  world  entertainment  essay
military           2831     5     3    16     9       4        8         10        0      15        8       24        9      3     6     42              6      1
baby                  0  2932     3     3    26       0        1          0       10       7       10        3       16      4     3      7             20      4
car                   6    10  2813     3     6       8       13          3        1      13       10        3       39      1    11      5             24      4
game                 10    11     6  2843     5       9        2          4        1      11       13        3        8      4    25      3             31      3
food                  0    38     0     3  2799       1        5          1       67      34       16        7        9      3     4      8             14     10
sports                2     7     6    13     6    2803        9          0        1      13       24        5       10      1     5     19             42      4
finance              12    10    13     4    15       6     2692          1        2      21        5        3       18      2    79     47             12      8
discovery             8     2     0     3     3       2        5       1155        1       5        1        1        1      0    13      9              0      1
regimen               0    59     0     0    63       0        2          0     1093       0        3        3        4      2     0      1              5      0
travel                9    19     8     8    23       4        9          8        0    2741       19       20       19      7    13     55             14     12
fashion               2    21     5     9    14       9        1          5       13      18     2772        5        7      1     6     11             77      7
history              49     9     2     3     6       3        3          6        4      28        3     2813       12     20     2     35             21      6
society              27    77    50     7    43       7       42          5       16      78       27       13     2414     29    36     36             58     15
story                 3    17     1     3     7       2        2          2        2       7        5       12       19   1120     4      6             14     11
tech                 16     8    19    21     6       3       52         13        3       6        5        4       14      0  2787      9             17      7
world                52    33    12     8     9      16       33         24        2      35       27       37       50      8    20   2583             30      4
entertainment         5    14     3    28     6      13        4          3        1       9      120       29       17      3    12     10           2708      8
essay                 7    23     5     3    12       1        8          6        4      15       22       11        7      2     5      2             11   1010
```

```python
    #get sub_result and Figure
>>> tc.plot_display(y_test, Y_predict)
>>> Plot display...
               Test count:  Predict count:  Sub Result:  Sub_Abs Result:
baby                  3049            3295          246              246
car                   2973            2949          -24               24
discovery             1210            1246           36               36
entertainment         2993            3104          111              111
essay                 1154            1115          -39               39
fashion               2983            3090          107              107
finance               2950            2891          -59               59
food                  3019            3058           39               39
game                  2992            2978          -14               14
history               3025            2996          -29               29
military              3000            3039           39               39
regimen               1235            1221          -14               14
society               2980            2673         -307              307
sports                2970            2891          -79               79
story                 1237            1210          -27               27
tech                  2990            3031           41               41
travel                2988            3056           68               68
world                 2983            2888          -95               95
```

Installation
----------------------
    $ pip install TextClassifier
