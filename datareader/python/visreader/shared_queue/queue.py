"""
# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import cPickle
import logging
import traceback
from sharedmemory import SharedMemoryMgr
from multiprocessing.queues import Queue

logger = logging.getLogger(__name__)


class SharedQueueError(ValueError):
    """ SharedQueueError
    """
    pass


class SharedQueue(Queue):
    """ a Queue based on shared memory to communicate data between Process,
        and it's interface is compatible with 'multiprocessing.queues.Queue'
    """

    def __init__(self, maxsize=0, mem_mgr=None, memsize=None, pagesize=None):
        """ init
        """
        super(SharedQueue, self).__init__(maxsize)
        if mem_mgr is not None:
            self._shared_mem = mem_mgr
        else:
            self._shared_mem = SharedMemoryMgr(
                capacity=memsize, pagesize=pagesize)

    def put(self, obj, **kwargs):
        """ put an object to this queue
        """
        obj = cPickle.dumps(obj, -1)
        buff = self._shared_mem.malloc(len(obj))
        buff.put(obj)

        try:
            super(SharedQueue, self).put(buff, **kwargs)
        except Exception as e:
            stack_info = traceback.format_exc()
            err_msg = 'failed to put a element to SharedQueue '\
                'with stack info[%s]' % (stack_info)
            logger.warn(err_msg)
            buff.free()
            raise e

    def get(self, **kwargs):
        """ get an object from this queue
        """
        buff = None
        try:
            buff = super(SharedQueue, self).get(**kwargs)
        except Exception as e:
            stack_info = traceback.format_exc()
            err_msg = 'failed to get element from SharedQueue '\
                        'with stack info[%s]' % (stack_info)
            logger.warn(err_msg)
            raise e

        data = buff.get()
        if data is None:
            raise SharedQueueError('failed to extract data from shared buffer[%s]'\
                % (str(buff)))

        buff.free()
        return cPickle.loads(data)

    def release(self):
        self._shared_mem.release()
        self._shared_mem = None
