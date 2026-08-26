#!/usr/bin/env python

import re
import numpy as np
import pandas as pd
import csv


'''
A module of useful tools to parse through stateScriptLog files
To use this module, import statescriptparse into your python code.

original author Kevin Fan
05/16/2016
updated Fall 2021 Alison Comrie
'''


'''
todataframe() is a generic stateScriptLog parsing function that converts any stateScriptLog into a DataFrame of
stateScript events.
Inputs:
log - stateScriptLog file
inputs - dictionary of important inputs, in the form of { input_channel_# : 'input_name' , etc}
outputs - dictionary of important outputs, in the form of { output_channel_# : 'output_name' , etc}
*** every important input and output must have a unique name
Output:
df - DataFrame of all important stateScript events (whenever either an important input or output or both changes state)
    This DataFrame has a column for every important input and output, labeled by name.
    This DataFrame has a row for every important stateScript event, labeled by timestamp.

'''


def todataframe(log, inputs, outputs):

    # create lists with input and output names and channel numbers
    input_names = []
    input_channels = []
    for i in inputs:
        input_channels.append(i)
        input_names.append(inputs[i])
    output_names = []
    output_channels = []
    for o in outputs:
        output_channels.append(o)
        output_names.append(outputs[o])

    # Regular expression objects created to aid in parsing the stateScriptLog
    ts_re = re.compile("(\d{1,20}) \d{1,10} \d{1,10}")  # Regular Expression for timestamp
    di_re = re.compile("\d{1,20} (\d{1,10}) \d{1,10}")  # RE for digital input
    do_re = re.compile("\d{1,20} \d{1,10} (\d{1,10})")  # RE for digital output

    # Go through stateScriptLog and find all events!
    ts = []
    di = []
    do = []
    for line in log:
        for match in re.finditer(ts_re, line):  # Grabs all TS
            ts.append(int(match.groups()[0]))
        for match in re.finditer(di_re, line):  # Grabs all DI
            di.append(int(match.groups()[0]))
        for match in re.finditer(do_re, line):  # Grabs all DO
            do.append(int(match.groups()[0]))
    if len(ts) == len(di) == len(do):  # Sanity check
        pass
    else:
        raise Exception("there's something terribly wrong with the stateScriptLog!!!")

    # Create raw data file
    input_len = len(inputs)
    output_len = len(outputs)
    rawdata = np.zeros((len(ts), input_len + output_len))

    for idx, inputstate in enumerate(di):  # Populates inputs
        inputstates = bin(inputstate)[2:]
        for bitnum, bitstate in enumerate(reversed(inputstates)):  # Bits must be read out in reverse
            for i, channel in enumerate(input_channels):
                if bitnum == channel:  # Bits are only important if they affect important inputs
                    rawdata[idx, i] = bitstate

    for idx, outputstate in enumerate(do):  # Populates outputs
        outputstates = bin(outputstate)[2:]
        for bitnum, bitstate in enumerate(reversed(outputstates)):  # Bits must be read out in reverse
            for i, channel in enumerate(output_channels):
                if bitnum == channel:  # Bits are only important if they affect important outputs
                    rawdata[idx, i + input_len] = bitstate  # Outputs are concatenated after inputs

    df = pd.DataFrame(rawdata, columns=input_names+output_names, index=ts)   # Generate Dataframe
    #df = df[df.sum(axis=1) != 0]
    return df


'''
smooth() is a function that smooths out the data from the input channel(s) to get rid of useless quick licking data.
The smoothing is simply done by switching a zero to a one if the zero is flanked by ones.
Input:
s - DataFrame series with values consisting only of ones and zeros
Output:
no output; function smooths the input series itself.
'''


def smooth(s):
    for i in range(len(s)):
        if i == 0:
            continue
        if i == len(s)-1:
            continue
        if s.iloc[i] == 0:
            if s.iloc[i-1] == s.iloc[i+1] == 1:
                s.iloc[i] = 1


'''
getstamp() is a function that grabs the timestamp of an interesting event, with a dataframe input.
Inputs:
s - pandas series with timestamps as indices
n - a value, typically 1 or 0, to look for in the series
param - a parameter, either 'first', 'last', 'after' or 'before' (default is 'first')
    - 'first' looks for the first time the value shows up in the series
    - 'last' looks for the last time the value shows up in the series
    - 'after' looks for the first time the value shows up in the series, after a specified timestamp
    - 'before' looks for the last time the value shows up in the series, before a specified timestamp
ind - optional input: specify an index(in series s) when using the 'after' or 'before' parameters.
Output: The index of found timestamp (in series s), or NaNs if timestamp doesn't exist.

'''


def getstamp(s, n, param, ind=float('nan')):
    iterator = []
    if param == 'first':
        # forward iterate through s from the top
        iterator = range(len(s))
    elif param == 'last':
        # reverse iterate through s from the bottom
        iterator = reversed(range(len(s)))
    else:
        if param == 'after':
            # forward iterate through s, starting at index ind. start timestamp is excluded
            iterator = range(ind+1, len(s))
        elif param == 'before':
            # reverse iterate through s starting at index ind. start timestamp is excluded
            iterator = reversed(range(ind))
    # iterate through the iterator, stopping when n is found for the first time.
    for i in iterator:
        if s.iloc[i] == n:
            return i
    # value wasn't found: return nan
    return float('nan')


'''
tocsvfile() is a function that writes all of your gathered per-trial metadata to a csv file.
Inputs:
f - a string specifying filename (ending in .csv) and its directory
*args - a variable number of arguments. Each argument specifies one row of data in csv file.
Outputs:
None, except written csv file

'''


def tocsvfile(f, *args):
    print('writing to %s ...' % f)
    writefile = open(f, 'wb')
    wr = csv.writer(writefile)
    [wr.writerow(arg) for arg in args]
    print('successfully wrote to file %s' % f)


'''
todffile() is a function that writes your dictionary into a dataframe and pickles it
*** Dan says pickling is not a good way of saving data because changes in python or pandas version could mess it up,
    and it's not a universal filetype ***
Inputs:
f - a string specifying pickle filename and its directory
d - your dictionary
drop - any indices (trials) you want to remove
Outputs:
the dataframe
'''


def todffile(f, d, drop=[]):
    print('making dataframe and pickling to %s ...' % f)
    df = pd.DataFrame(d)
    if drop:
        df = df.drop(df.index[drop])
    df.to_pickle(f)
    print('successfully wrote to %s' % f)
    return df


'''
todf() just turns dictionary to dataframe ezpz
'''


def todf(d):
    return pd.DataFrame(d)