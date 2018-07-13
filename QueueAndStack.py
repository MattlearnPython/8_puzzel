#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 20:31:58 2018

@author: jinzhao
"""

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    
    def get_contents(self):
        return self.items
    
class Stack: 
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enstack(self, item):
        self.items.append(item)

    def destack(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    
    def get_contents(self):
        return self.items

