"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


# constraints are in conjunctive normal form (CNF)

import re

def clean_string(string):
    return string.replace('\n', '')

def label_index(label):
    return int(clean_string(label)[1:])

def get_constraints_for_dominant_label(label, failed_labels):
    and_list = []
    for other in failed_labels:
        if other != label:
            and_list.append([(label, other, 0)])
    return and_list

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def get_constraints_from_file(file):
    and_list = []
    lines = open(file, 'r').readlines()  # AND

    num_labels = int(lines[0])
    for index in range(1, len(lines)):
        elements = re.split(' +', lines[index])
        i = 0
        labels = []  # OR
        while elements[i].startswith('y'):
            labels.append(label_index(elements[i]))
            i += 1

        constraint = clean_string(elements[i])
        i += 1

        if constraint == 'min':
            for other in range(num_labels):
                if other not in labels:
                    and_list.append([(other, label, 0) for label in labels])

        if constraint == 'max':
            for other in range(num_labels):
                if other not in labels:
                    and_list.append([(label, other, 0) for label in labels])

        if constraint == 'notmin':
            others = filter(lambda x: x not in labels, range(num_labels))
            # this constraint makes only sense with one label
            label = labels[0]
            and_list.append([(label, other, 0) for other in others])

        if constraint == 'notmax':
            others = filter(lambda x: x not in labels, range(num_labels))
            # this constraint makes only sense with one label
            label = labels[0]
            and_list.append([(other, label, 0) for other in others])

        if constraint == '<':
            label2 = label_index(elements[i])
            and_list.append([(label2, label, 0) for label in labels])

        if constraint == '>':
            label2 = label_index(elements[i])
            and_list.append([(label, label2, 0) for label in labels])
        if constraint == '<=' and isfloat(elements[i]):
            and_list.append([(label, -1, float(elements[i])) for label in labels])

    return and_list
