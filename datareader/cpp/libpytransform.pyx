#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector as vector
from libcpp.map cimport map
from libcpp.pair cimport pair as pair
from libcpp cimport bool
from libc.string cimport memcpy
from libc.stdint cimport uintptr_t
import time
import numpy as np
from cython.operator cimport dereference as deref, preincrement as inc

def fmt_time():
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

import logging
logger = logging.getLogger(__name__)
cdef extern from "transformer.h" namespace "vistool":
    cdef struct transformer_input_data_t:
        unsigned int id
        string data
        string label
        
    cdef struct transformer_output_data_t:
        unsigned int id
        int err_no
        string err_msg
        string label
        vector[int] shape
        string data
        
cdef extern from "baseprocess.h" namespace "vistool":
    cdef cppclass IProcessor:
        int process(const transformer_input_data_t &input,\
             transformer_output_data_t &output) nogil

    IProcessor *create_processor "vistool::IProcessor::create" \
        (const string &classname, const vector[map[string, string]] &ops)
    void destroy_processor "vistool::IProcessor::destroy" (IProcessor *p) nogil


class TransformerException(Exception):
    pass
 

cdef class CyProcessor:
    cdef IProcessor *_cprocessor
    cdef string classname
    cdef map[string, string] conf
    cdef vector[map[string, string]] ops_conf
    def __cinit__(self, ops_conf):
        if len(ops_conf) == 1 and ops_conf[0]['op_name'] == 'lua_op':
            self.classname = 'LuacvProcess'
        else:
            self.classname = 'ImageProcess'

        self.ops_conf = ops_conf
        self._cprocessor = create_processor(self.classname, self.ops_conf)

        if self._cprocessor is NULL:
            raise TransformerException('fail to create a IProcessor(%s)' % (self.classname))

        logger.debug('create %s' % (str(self)))

    def get_conf(self):
        return self.classname, self.ops_conf

    def process(self, data, label, id=None):
        cdef transformer_input_data_t input
        cdef transformer_output_data_t output

        input.data = data
        input.label = label

        if id is not None:
            input.id = id
        else:
            input.id = 0

        cdef int ret = 0
        with nogil:
            ret = self._cprocessor.process(input, output)

        if ret != 0:
            raise TransformerException('failed to process with ret[%d]' % (ret))

        shape = list(output.shape)
        out_data = np.fromstring(output.data, np.uint8).reshape(shape)
        out_label = output.label
        return out_data, out_label

    def __str__(self):
        """ representation for this object
        """
        addr = 0
        if self._cprocessor != NULL:
            addr = <uintptr_t>self._cprocessor

        return 'CyProcessor(0x%x)' % (addr)

    def __dealloc__(self):
        if self._cprocessor != NULL:
            logger.debug('destroy %s' % (str(self)))
            with nogil:
                destroy_processor(self._cprocessor)

            self._cprocessor = NULL


cdef extern from "transformer.h" namespace "vistool":
    cdef cppclass Transformer:
        int init(const map[string,string] &conf)
        int set_processor(IProcessor *processor)
        int start() nogil
        int stop() nogil
        bool is_stopped()
        int put(const transformer_input_data_t &input) nogil
        int put(int id, const char *image, int image_len,
            const char *label, int label_len) nogil
        int get(transformer_output_data_t *output) nogil

       
cdef extern from "transformer.h" namespace "vistool":
    Transformer *create_transform "vistool::Transformer::create" (const string &type)
    void destroy_transform "vistool::Transformer::destroy" (Transformer *t) nogil


cdef class CyTransformer:
    cdef Transformer *_ctransformer
    cdef unsigned int id
    def __cinit__(self, trans_conf, ops_conf):
        for op in ops_conf:
            for k, v in op.items():
                op[str(k)] = str(v)
        for k, v in trans_conf.items():
            trans_conf[str(k)] = str(v)

        self.id = 0
        ctransformtype = 'ImageTransformer'
        self._ctransformer = create_transform(ctransformtype)
        if self._ctransformer is NULL:
            raise TransformerException('fail to create a transformer')

        cdef string c_name = 'ImageProcess'
        if len(ops_conf) == 1 and ops_conf[0]['op_name'] == 'lua_op':
            c_name = 'LuacvProcess'
            #  notes: we need allocate lua states for each thread
            if 'state_num' not in ops_conf[0]:
                ops_conf[0]['state_num'] = trans_conf['thread_num']

        cdef vector[map[string, string]] c_ops_conf = ops_conf
        cdef IProcessor *proc_ptr = create_processor(c_name, c_ops_conf)
        self._ctransformer.set_processor(proc_ptr)
        logger.debug('create %s' % (str(self)))
        
    def __init__(self, trans_conf, ops_conf):
        cdef map[string,string] cconf
        for k, v in trans_conf.items():
            cconf[str(k)] = str(v)
        cdef int r = 0
        r = self._ctransformer.init(cconf)
        if r < 0:
            raise TransformerException('fail to call init with ret[%d]' % (r))

    def start(self):
        cdef int r = 0
        with nogil:
            r = self._ctransformer.start()

        if r < 0:
            raise TransformerException('fail to call start with ret[%d]' % (r))
            
    def stop(self):
        cdef int r = 0

        with nogil:
            r = self._ctransformer.stop()

        return r == 0
       
    def is_stopped(self):
        cdef bool r 
        r = self._ctransformer.is_stopped()
        return r
    
    def put(self, image, label, context):
        self.id += 1
        cdef const char * imagedata = <const char*>image
        cdef int image_len = len(image)
        cdef const char * labeldata = <const char*>label
        cdef int label_len = len(label)

        cdef int id = self.id
        if context is not None and 'id' in context:
            id = context['id']

        cdef int r = 0
        cdef Transformer *ctransformer = self._ctransformer
        with nogil:
            r = ctransformer.put(id, imagedata, image_len, labeldata, label_len)
    
        if r < 0:
            raise TransformerException('fail to put data to transformer with ret[%d]' % (r))
            
    def get(self, context):
        cdef transformer_output_data_t data
        cdef int r = 0

        cdef Transformer *ctransformer = self._ctransformer
        with nogil:
            r = ctransformer.get(&data)

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
            outputarray = np.frombuffer(data.data, np.uint8).reshape(shape)
            return outputarray, str(data.label)
        else:
            if context is not None:
                context['req_data'] = data.data
                context['req_label'] = str(data.label)
            raise TransformerException('failed to transform image with')

    def __str__(self):
        """ representation for this object
        """
        addr = 0
        if self._ctransformer != NULL:
            addr = <uintptr_t>self._ctransformer

        return 'CyTransformer(0x%x)' % (addr)

    def __dealloc__(self):
        if self._ctransformer != NULL:
            logger.debug('destroy %s' % (str(self)))
            with nogil:
                destroy_transform(self._ctransformer)

            self._ctransformer = NULL
