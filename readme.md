# Crystal Ball

This python project creates a generic neural network that takes arbitrary data to calculate patterns. It uses typical neural network stuff (neurons, backpropagation, etc.) and the training data is open to any data.

Here is a training data example:

```
training_data = [
    [{'cloudy': 0, 'windy': 0}, None],
    [{'cloudy': 1, 'windy': 0}, 'umbrella'],
    [{'cloudy': 0, 'windy': 1}, None],
    [{'cloudy': 1, 'windy': 1}, None],
]
```

This should tell the network to only bring an umbrella if it is cloudy but not windy.

If we run `python3 umbrella_wind.py`, the network trains itself and produces:

```
PREDICT TRAINED
          conn. group umbrella, b-2.692648, cst+0.004032
cloudy   (+0.00) ---------- w+5.214098 ----------> umbrella (+0.0634085805)
windy    (+0.00) ---------- w-5.472915 ----------> umbrella (+0.0634085805)
          conn. group umbrella, b-2.692648, cst+0.004032
cloudy   (+1.00) ---------- w+5.214098 ----------> umbrella (+0.9256319442)
windy    (+0.00) ---------- w-5.472915 ----------> umbrella (+0.9256319442)
          conn. group umbrella, b-2.692648, cst+0.004032
cloudy   (+0.00) ---------- w+5.214098 ----------> umbrella (+0.0002841959)
windy    (+1.00) ---------- w-5.472915 ----------> umbrella (+0.0002841959)
          conn. group umbrella, b-2.692648, cst+0.004032
cloudy   (+1.00) ---------- w+5.214098 ----------> umbrella (+0.0496673377)
windy    (+1.00) ---------- w-5.472915 ----------> umbrella (+0.0496673377)
```

We see that if cloudy=1 (true) and windy=0 (false), the system predicts to bring an umbrella with ~92% certainty.

We can do the same for programming language prediction:

When we train the system with data to predict javascript or python like this:

```
training_data = [
    [{'def': 2, 'print': 1, 'None': 2, 'for': 3}, 'python'],
    [{'def': 1, 'print': 2, 'None': 3, 'for': 4}, 'python'],
    [{'print': 1, 'for': 1}, 'python'],
    [{'def': 8}, 'python'],
    [{'function': 8, 'forEach': 1, ';': 6}, 'javascript'],
    [{'function': 1, 'forEach': 5, ';': 3, 'this': 7}, 'javascript'],
    [{'function': 3, 'forEach': 1, 'this': 1}, 'javascript'],
    [{'function': 2, 'forEach': 2, 'def': 2, ';': 7, 'this': 3}, 'javascript'],
]
```

And we run `python3 code_detector.py`, we get:

```
Predict [{'function': 2, 'forEach': 2, 'def': 2, ';': 7, 'this': 3}, 'javascript']
          conn. group python, b+1.689360, cst+0.000076
def      (+2.00) ---------- w+0.593196 ----------> python (+0.0010028144)
print    (+0.00) ---------- w-0.097580 ----------> python (+0.0010028144)
None     (+0.00) ---------- w-1.091604 ----------> python (+0.0010028144)
for      (+0.00) ---------- w+3.142515 ----------> python (+0.0010028144)
function (+2.00) ---------- w-1.466844 ----------> python (+0.0010028144)
forEach  (+2.00) ---------- w-1.863424 ----------> python (+0.0010028144)
;        (+7.00) ---------- w-0.111835 ----------> python (+0.0010028144)
this     (+3.00) ---------- w-0.778770 ----------> python (+0.0010028144)
          conn. group javascript, b-0.798999, cst+0.000068
def      (+2.00) ---------- w-0.706417 ----------> javascript (+0.9999991047)
print    (+0.00) ---------- w-2.267551 ----------> javascript (+0.9999991047)
None     (+0.00) ---------- w+0.447324 ----------> javascript (+0.9999991047)
for      (+0.00) ---------- w-1.721714 ----------> javascript (+0.9999991047)
function (+2.00) ---------- w+2.080340 ----------> javascript (+0.9999991047)
forEach  (+2.00) ---------- w-0.319708 ----------> javascript (+0.9999991047)
;        (+7.00) ---------- w+1.661140 ----------> javascript (+0.9999991047)
this     (+3.00) ---------- w+0.329559 ----------> javascript (+0.9999991047)
```

We see that if we feed the system the words {'function': 2, 'forEach': 2, 'def': 2, ';': 7, 'this': 3}, it thinks this is javascript with a 99% certainty.

If you want to test it yourself, start by checking out umbrella.py. This uses just one input neuron and one output neuron.

## How to make this work?

- Clone the project
- Create a [virtual environment](https://www.justgivemeanexample.com/example/create-a-virtual-environment-in-python)
- [Activate virtual environment](https://www.justgivemeanexample.com/example/activate-python-virtual-environment-in-bash)
- [Install requirements](https://www.justgivemeanexample.com/example/install-pip-packages-from-requirementstxt-in-bash) (Numpy)
- Run code: `python3 umbrella.py`  
