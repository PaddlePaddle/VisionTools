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
""" utils for memory management which is allocated on sharedmemory,
    note that all this structures are not thread-safe or process-safe
"""
import os
import math
import struct
import ctypes
import weakref
import logging
from multiprocessing import Lock
from multiprocessing import RawArray

logger = logging.getLogger(__name__)


class SharedMemoryError(ValueError):
    """ SharedMemoryError
    """
    pass


class SharedBufferError(SharedMemoryError):
    """ SharedBufferError
    """
    pass


class SharedBuffer(object):
    """ Buffer allocated from SharedMemoryMgr, and it stores data on shared memory

        note that: 
            every instance of this should be freed explicitely by calling 'self.free'
    """

    def __init__(self, owner, capacity, pos, size=0, alloc_status=''):
        """ init
        """
        self._owner = owner
        self._cap = capacity  # capacity in bytes
        self._pos = pos  # page position
        self._size = size  # used bytes
        self._alloc_status = alloc_status
        assert self._pos >= 0 and self._cap > 0, \
            "invalid params[%d:%d] to construct SharedBuffer" \
            % (self._pos, self._cap)

    def owner(self):
        """ get owner
        """
        return SharedMemoryMgr.get_mgr(self._owner)

    def put(self, data, override=False):
        """ put data to this buffer

        Args:
            @data (str): data to be stored in this buffer

        Returns:
            None

        Raises:
            SharedMemoryError when not enough space in this buffer
        """
        assert type(data) == str, 'invalid type for SharedBuffer::put'
        if self._size > 0 and not override:
            raise SharedBufferError('already has already been setted before')

        if self.capacity() < len(data):
            raise SharedBufferError('data[%d] is larger than size of buffer[%s]'\
                % (len(data), str(self)))

        self.owner().put_data(self, data)
        self._size = len(data)

    def get(self, offset=0, size=None):
        """ get the data stored this buffer

        Args:
            @offset (int): position for the start point to 'get'
            @size (int): size to get

        Returns:
            data (str): user's data passed in by 'put' if exist
            None: if no data stored in
        """
        offset = offset if offset >= 0 else self._size + offset
        if self._size <= 0:
            return None

        size = self._size if size is None else size
        assert offset + size <= self._cap, 'invalid offset[%d] '\
            'or size[%d] for capacity[%d]' % (offset, size, self._cap)
        return self.owner().get_data(self, offset, size)

    def size(self):
        """ bytes of used memory
        """
        return self._size

    def resize(self, size):
        """ resize the used memory to 'size', should not be greater than capacity
        """
        assert size >= 0 and size <= self._cap, \
            "invalid size[%d] for resize" % (size)

        self._size = size

    def capacity(self):
        """ size of allocated memory
        """
        return self._cap

    def __str__(self):
        """ human readable format
        """
        return "SharedBuffer(owner:%s, pos:%d, size:%d, "\
            "capacity:%d, alloc_status:[%s], pid:%d)" \
            % (str(self._owner), self._pos, self._size, \
            self._cap, self._alloc_status, os.getpid())

    def free(self):
        """ free this buffer to it's owner
        """
        if self._owner is not None:
            self.owner().free(self)
            self._owner = None
            self._cap = 0
            self._pos = -1
            self._size = 0
            return True
        else:
            return False


class PageAllocator(object):
    """ allocator used to malloc and free shared memory which
        is split into pages
    """
    s_header_magic = 1234321
    s_allocator_header = 12

    def __init__(self, base, total_pages, page_size):
        """ init
        """
        self._base = base
        self._total_pages = total_pages
        self._page_size = page_size

        header_pages = int(math.ceil(float(total_pages + \
            self.s_allocator_header) / page_size))
        self._header_pages = header_pages
        self._free_pages = total_pages - header_pages
        self._header_size = self._header_pages * page_size
        self._reset()

    def _reset(self):
        alloc_page_pos = self._header_pages
        used_pages = self._header_pages
        header_info = struct.pack('III', \
            self.s_header_magic, alloc_page_pos, used_pages)
        assert len(header_info) == self.s_allocator_header, \
            'invalid size of header_info'

        self._base[0:self.s_allocator_header] = header_info
        self.set_page_status(0, self._header_pages, '1')
        self.set_page_status(self._header_pages, self._free_pages, '0')

    def header(self):
        """ get header info of this allocator
        """
        magic, pos, used = struct.unpack('III',
                                         self._base[0:self.s_allocator_header])
        assert magic == self.s_header_magic, \
            'invalid header magic[%d] in shared memory' % (magic)
        return self._header_pages, self._total_pages, pos, used

    def empty(self):
        """ are all allocatable pages available
        """
        header_pages, pages, pos, used = self.header()
        return header_pages == used

    def full(self):
        """ are all allocatable pages used
        """
        header_pages, pages, pos, used = self.header()
        return header_pages + used == pages

    def __str__(self):
        header_pages, pages, pos, used = self.header()
        desc = '{page_info[total:%d,used:%d,header:%d,alloc_pos:%d,pagesize:%d]}' \
            % (pages, used, header_pages, pos, self._page_size)
        return 'PageAllocator:%s' % (desc)

    def set_alloc_info(self, alloc_pos, used_pages):
        """ set allocating position to new value
        """
        self._base[4:12] = struct.pack('II', alloc_pos, used_pages)

    def set_page_status(self, start, page_num, status):
        """ set pages from 'start' to 'end' with new same status 'status'
        """
        assert status in ['0', '1'], 'invalid status[%s] for page status '\
            'in allocator[%s]' % (status, str(self))
        start += self.s_allocator_header
        end = start + page_num
        assert start >= 0 and end <= self._header_size, 'invalid end[%d] of pages '\
            'in allocator[%s]' % (end, str(self))
        self._base[start:end] = status * page_num

    def get_page_status(self, start, page_num):
        start += self.s_allocator_header
        end = start + page_num
        assert start >= 0 and end <= self._header_size, 'invalid end[%d] of pages '\
            'in allocator[%s]' % (end, str(self))
        status = self._base[start:end]
        zero_num = status.count('0')
        if zero_num == 0:
            return (page_num, 1)
        else:
            return (zero_num, 0)

    def malloc_page(self, page_num):
        header_pages, pages, pos, used = self.header()
        end = pos + page_num
        if end > pages:
            pos = self._header_pages
            end = pos + page_num

        page_status = self.get_page_status(pos, page_num)
        if page_status != (page_num, 0):
            free_pages = self._total_pages - used
            if free_pages == 0:
                err_msg = 'all memory pages have been used:%s' % (str(self))
            else:
                err_msg = 'not found available pages with page_status[%s] '\
                    'and %d free pages' % (str(page_status), free_pages)
            err_msg = 'failed to malloc %d pages at pos[%d] for reason[%s] and allocator status[%s]' \
                % (page_num, pos, err_msg, str(self))
            logger.warn(err_msg)
            raise SharedMemoryError(err_msg)

        self.set_page_status(pos, page_num, '1')
        used += page_num
        self.set_alloc_info(end, used)

        assert self.get_page_status(pos, page_num) == (page_num, 1), \
            'faild to validate the page status'
        return pos

    def free_page(self, start, page_num):
        """ free 'page_num' pages start from 'start'
        """
        page_status = self.get_page_status(start, page_num)
        assert page_status == (page_num, 1), \
            'invalid status[%s] when free [%d, %d]' \
                % (str(page_status), start, page_num)
        self.set_page_status(start, page_num, '0')
        _, _, pos, used = self.header()
        used -= page_num
        self.set_alloc_info(pos, used)


DEFAULT_SHARED_MEMORY_SIZE = 1024 * 1024 * 1024


class SharedMemoryMgr(object):
    """ manage a continouse block of memory, provide
        'malloc' to allocate new buffer, and 'free' to free buffer
    """
    s_memory_mgrs = weakref.WeakValueDictionary()
    s_mgr_num = 0
    s_log_statis = False

    @classmethod
    def get_mgr(cls, id):
        """ get a SharedMemoryMgr with size of 'capacity'
        """
        assert id in cls.s_memory_mgrs, 'invalid id[%s] for memory managers' % (
            id)
        return cls.s_memory_mgrs[id]

    def __init__(self, capacity=None, pagesize=None):
        """ init
        """
        logger.debug('create SharedMemoryMgr')

        pagesize = 64 * 1024 if pagesize is None else pagesize
        assert type(pagesize) is int, "invalid type of pagesize[%s]" \
            % (str(pagesize))

        capacity = DEFAULT_SHARED_MEMORY_SIZE if capacity is None else capacity
        assert type(capacity) is int, "invalid type of capacity[%s]" \
            % (str(capacity))

        assert capacity > 0, '"size of shared memory should be greater than 0'
        self._released = False
        self._cap = capacity
        self._page_size = pagesize

        assert self._cap % self._page_size == 0, \
            "capacity[%d] and pagesize[%d] are not consistent" \
            % (self._cap, self._page_size)
        self._total_pages = int(self._cap / self._page_size)

        self._pid = os.getpid()
        SharedMemoryMgr.s_mgr_num += 1
        self._id = self._pid * 100 + SharedMemoryMgr.s_mgr_num
        SharedMemoryMgr.s_memory_mgrs[self._id] = self
        self._locker = Lock()
        self._setup()

    def _setup(self):
        self._shared_mem = RawArray('c', self._cap)
        self._base = ctypes.addressof(self._shared_mem)
        self._locker.acquire()
        try:
            self._allocator = PageAllocator(self._shared_mem,
                                            self._total_pages, self._page_size)
        finally:
            self._locker.release()

    def malloc(self, size):
        """ malloc a new SharedBuffer

        Args:
            @size (int): buffer size to be malloc

        Returns:
            SharedBuffer

        Raises:
            SharedMemoryError when not found available memory
        """
        page_num = int(math.ceil(float(size) / self._page_size))
        size = page_num * self._page_size

        start = None
        self._locker.acquire()
        try:
            start = self._allocator.malloc_page(page_num)
            alloc_status = str(self._allocator)
        finally:
            self._locker.release()

        if start is None:
            raise SharedMemoryError('failed to malloc %d bytes of memory' %
                                    (size))

        return SharedBuffer(self._id, size, start, alloc_status=alloc_status)

    def free(self, shared_buf):
        """ free a SharedBuffer

        Args:
            @shared_buf (SharedBuffer): buffer to be freed

        Returns:
            None

        Raises:
            SharedMemoryError when failed to free
        """
        assert shared_buf._owner == self._id, "invalid shared_buf[%s] "\
            "for it's not allocated from me[%s]" % (str(shared_buf), str(self))
        cap = shared_buf.capacity()
        start_page = shared_buf._pos

        page_num = cap / self._page_size
        assert page_num * self._page_size == cap, "invalid capacity "\
            "of shared_buf[%s] when free it" % (str(shared_buf))

        #maybe we don't need this lock here
        self._locker.acquire()
        try:
            self._allocator.free_page(start_page, page_num)
        finally:
            self._locker.release()

    def put_data(self, shared_buf, data):
        """  fill 'data' into 'shared_buf'
        """
        assert len(data) <= shared_buf.capacity(), 'too large data[%d] '\
            'for this buffer[%s]' % (len(data), str(shared_buf))
        start = shared_buf._pos * self._page_size
        assert start >= 0 and start + len(data) < self._cap, "invalid start "\
            "position[%d] when put data to buff:%s" % (start, str(shared_buf))
        ctypes.memmove(self._base + start, data, len(data))

    def get_data(self, shared_buf, offset, size):
        """ extract 'data' from 'shared_buf' in range [offset, offset + size)
        """
        start = shared_buf._pos * self._page_size
        start += offset
        return self._shared_mem[start:start + size]

    def release(self):
        """ called when all processes will not use this memory in future
        """
        self._released = True

    def __str__(self):
        return 'SharedMemoryMgr:{id:%d, %s}' % (self._id, str(self._allocator))

    def __del__(self):
        if SharedMemoryMgr.s_log_statis:
            logger.info('destroy [%s]' % (self))

        if not self._released and not self._allocator.empty():
            logger.warn('not empty when delete this SharedMemoryMgr[%s]' %
                        (self))

        if self._id in SharedMemoryMgr.s_memory_mgrs:
            del SharedMemoryMgr.s_memory_mgrs[mgr._id]
            SharedMemoryMgr.s_mgr_num -= 1
