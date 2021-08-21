#!/usr/bin/env python

def tar_split(targets,idx): 
    # argets: a list includes 4 targets
    # idx: which target [0,1,2,3]
    return [tar[idx] for tar in targets]
     


