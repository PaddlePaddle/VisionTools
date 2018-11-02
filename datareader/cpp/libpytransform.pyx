#!/usr/bin/env python
# -*- coding: utf-8 -*-

from libcpp cimport bool
#from opencv cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector as vector
from libcpp.map cimport map
from libcpp.pair cimport pair as pair
from libcpp cimport bool
from libc.string cimport memcpy
import time
import numpy as np
from cython.operator cimport dereference as deref, preincrement as inc

def fmt_time():
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

cdef extern from "transformer.h" namespace "vis":
    #int maintools(int argc, char * argv[])
    cdef struct transformer_input_data_t:
        unsigned int id
        vector[char] data
        string label
        
    cdef struct transformer_output_data_t:
        unsigned int id
        int err_no
        string err_msg
        string data
        vector[int] shape
        string label
        
    cdef cppclass Transformer:
        int init(const map[string,string] &conf)
        int add_op(const string &op_name, const map[string,string]&conf)
        int start() nogil
        int stop() nogil
        bool is_stopped()
        int put(const transformer_input_data_t &input) nogil
        int get(transformer_output_data_t &output) nogil
        
cdef extern from "transformer.h" namespace "vis":
    Transformer *create_transform "vis::Transformer::create" (const string &type)
    void destroy_transform "vis::Transformer::destroy" (Transformer *t) nogil

class TransformerException(Exception):
    pass
    
cdef class pyTransformer:
    cdef Transformer * thisptr
    cdef unsigned int id
    def __cinit__(self, conf, transformtype=None):
        if transformtype is None:
            transformtype = 'ImageTransformer'

        if type(transformtype) != str or type(conf) != dict:
            raise TransformerException('invalid param for Transformer')

        self.id = 0
        self.thisptr = create_transform(transformtype)
        if self.thisptr is NULL:
            raise TransformerException('fail to create a transformer')
        
    def __init__(self, conf, transformtype=None):
        cdef map[string,string] cconf
        for key, value in conf.iteritems():
            cconf[str(key)] = str(value)
        cdef int r = 0
        r = self.thisptr.init(cconf)
        if r < 0:
            raise TransformerException('fail to call init')
            
    def add_op(self, op_name, conf):
        cdef map[string,string] cconf
        for key, value in conf.iteritems():
            cconf[str(key)] = str(value)

        cdef int r = 0
        r = self.thisptr.add_op(op_name, cconf)
        if r < 0:
            raise TransformerException('fail to call add_op')

    def start(self):
        cdef int r = 0
        with nogil:
            r = self.thisptr.start()

        if r < 0:
            raise TransformerException('fail to call start')
            
    def stop(self):
        cdef int r = 0

        with nogil:
            r = self.thisptr.stop()

        return r == 0
       
    def is_stopped(self):
        cdef bool r 
        r = self.thisptr.is_stopped()
        return r
    
    def put(self, image, label, context):
        cdef transformer_input_data_t data
        self.id += 1
        data.id = self.id
        data.label = label
        data.data.resize(len(image))
        cdef const char * imagedata = <const char*>image
        cdef char * inputdata = &data.data[0]
        memcpy(inputdata, imagedata, len(image))

        if context is not None and 'id' in context:
            data.id = context['id']

        cdef int r = 0
        cdef Transformer *thisptr = self.thisptr
        with nogil:
            r = thisptr.put(data)
    
        if r < 0:
            raise TransformerException('fail to put data to transformer with ret[%d]' % (r))
            
    def get(self, context):
        cdef transformer_output_data_t data
        cdef int r = 0
        cdef Transformer * thisptr = self.thisptr

        with nogil:
            r = thisptr.get(data)

        if r != 0:
            return None, None

        if context is not None:
            context['retcode'] = r
            context['id'] = data.id
            context['err_no'] = data.err_no
            context['err_msg'] = data.err_msg

        outputarray = data.data
        if data.err_no == 0:
            shape = list(data.shape)
            outputarray = np.fromstring(data.data, np.uint8).reshape(shape)
            return outputarray, str(data.label)
        else:
            if context is not None:
                context['req_data'] = data.data
                context['req_label'] = str(data.label)
            raise TransformerException('failed to transform image with')

    def __dealloc__(self):
        if self.thisptr != NULL:
            with nogil:
                destroy_transform(self.thisptr)

            self.thisptr = NULL
