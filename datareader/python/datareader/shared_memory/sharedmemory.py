""" utils for memory management which is allocated on sharedmemory,
    note that all this structures are not thread-safe or process-safe
"""
import os
import ctypes
import logging

logger = logging.getLogger(__name__)


class SharedMemoryError(ValueError):
    """ SharedMemoryError
    """
    pass


class SharedBufferError(SharedMemoryError):
    """ SharedBufferError
    """
    pass


class AuthorityCheckError(SharedMemoryError):
    """ AuthorityCheckError
    """
    pass


class SharedBuffer(object):
    """ Buffer allocated from SharedMemoryMgr, and it stores data 
        on shared memory
        note that: 
            every instance of this should be freed explicitely by 
            the same process which is also the producer of this buffer
    """

    def __init__(self, owner, capacity, pos, size=0):
        """ init
        """
        self._owner = owner
        self._cap = capacity
        self._pos = pos
        self._size = size
        assert self._pos >= 0 and self._cap > 0, \
            "invalid params[%d:%d] to construct SharedBuffer" \
            % (self._pos, self._cap)

    def dump(self):
        """ get a representation of this object
        """
        info = (self._owner, self._cap, self._pos, self._size)
        return info

    @classmethod
    def load(cls, info):
        """ load and restore a SharedBuffer using 'info'
        """
        return SharedBuffer(*info)

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
        return self.owner().get_data(self, self._pos + offset, size)

    def size(self):
        """ size of used memory
        """
        return self._size

    def resize(self, size):
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
        return "SharedBuffer(owner:%s, pos:%d, size:%d, capacity:%d, pid:%d)" \
            % (str(self._owner), self._pos, self._size, self._cap, os.getpid())

    def free(self):
        """ free this buffer to it's owner
        """
        if self._owner is not None and self.owner().freeable(self):
            self.owner().free(self)
            self._owner = None
            self._cap = 0
            self._pos = -1
            self._size = 0
            return True
        else:
            return False


def authority_checked(action):
    """ check whether this process whether has the authority
        to do this action
    """

    def _safe_action(self, *args, **kwargs):
        pid = os.getpid()
        if pid != self._pid:
            raise AuthorityCheckError('check authority failed'\
                '[caller:%d is not owner:%d]' % (pid, self._pid))

        return action(self, *args, **kwargs)

    return _safe_action


class SharedMemoryMgr(object):
    """ manage a continouse block of memory, provide
        'malloc' to allocate new buffer, and 'free' to free buffer
    """
    s_memory_mgrs = {}
    s_mgr_num = 0
    s_log_statis = False

    @classmethod
    def get_mgr(cls, id):
        """ get a SharedMemoryMgr with size of 'capacity'
        """
        assert id in cls.s_memory_mgrs, 'invalid id[%s] for memory managers' % (
            id)
        return cls.s_memory_mgrs[id]

    def __init__(self, capacity=1 * 1024 * 1024, alignment=1024):
        """ init
        """
        logger.debug('create SharedMemoryMgr')
        assert capacity > 0, '"size of shared memory should be greater than 0'
        self._cap = capacity
        self._alignment = alignment
        self._pid = os.getpid()
        self.s_mgr_num += 1
        self._id = self._pid * 100 + self.s_mgr_num
        self.s_memory_mgrs[self._id] = self
        self._statis = {'id': self._id, 'free_times': 0, \
            'malloc_times': 0, 'used': 0, 'capacity': self._cap}
        self._setup()

    def _setup(self):
        from multiprocessing import RawArray

        self._shared_mem = RawArray('c', self._cap)
        self._base = ctypes.addressof(self._shared_mem)
        self._allocating_block = [0, self._cap]  # [(pos, cap)]
        # key is start-point or end-point of freed range
        # value is a tuple (start_pos, start_pos + cap) 
        self._freed_blocks = {}

    def _extend_block(self, size):
        """ extend the allocating block with more freed space to the tail
            if not reach the limit of 'self._cap', otherwise begins from
            the 0

        Args:
            @size (int): buffer size caller wants to be extended to

        Returns:
            True if succeed, False otherwise
        """
        pos, cap = self._allocating_block
        end_pos = pos + cap
        fblocks = self._freed_blocks
        # put back this allocating block to freed blocks if
        # it's end meets the limits
        if end_pos == self._cap:
            if pos != end_pos:
                fblocks[pos] = [pos, end_pos]
                fblocks[end_pos] = fblocks[pos]
            pos = 0
            cap = 0
            end_pos = 0

        if end_pos in fblocks:
            head, tail = fblocks[end_pos]
            assert head == end_pos, 'invalid head in freed blocks'
            del fblocks[head]
            del fblocks[tail]
            self._allocating_block = [pos, tail - pos]

        pos, cap = self._allocating_block
        if cap < size:
            return False
        else:
            return True

    @authority_checked
    def malloc(self, size, no_exp=False):
        """ malloc a new SharedBuffer

        Args:
            @size (int): buffer size to be malloc

        Returns:
            SharedBuffer

        Raises:
            SharedMemoryError when not found available memory
        """
        alignment = self._alignment
        if size % alignment != 0:
            size = (int(size / alignment) + 1) * alignment

        if self._allocating_block[1] < size:
            if not self._extend_block(size):
                statis = self.get_statis()
                assert self._statis['used'] <= self._cap, \
                    'FATAL: memory leak in shared memory[%s]' % (statis)
                logger.warn('WARN: failed to reclaim spare memory[%s]' %
                            (statis))
                if no_exp:
                    return None
                else:
                    raise SharedMemoryError('no spare space left '\
                        'for this malloc[size:%d][statis:%s]' % (size, statis))

        pos, cap = self._allocating_block
        self._allocating_block = [pos + size, cap - size]

        self._statis['malloc_times'] += 1
        self._statis['used'] += size
        return SharedBuffer(self._id, size, pos)

    def freeable(self, shared_buf):
        """ whether this manager can free this shared_buf
        """
        pid = os.getpid()
        if pid == self._pid and shared_buf.capacity() > 0:
            return True
        else:
            return False

    @authority_checked
    def free(self, shared_buf):
        """ free a SharedBuffer 'shared_buf'

        Args:
            @shared_buf (SharedBuffer): buffer to be freed

        Returns:
            None

        Raises:
            SharedMemoryError when failed to free
        """
        assert shared_buf._owner == self._id, "invalid shared_buf "\
            "for it's not allocated from me"
        size = shared_buf.capacity()
        start_pos = shared_buf._pos
        end_pos = start_pos + shared_buf.capacity()
        fblocks = self._freed_blocks

        prev_block = None
        next_block = None
        if start_pos in fblocks:
            head, tail = fblocks[start_pos]
            assert tail == start_pos, 'fatal:invalid tail pos in freed blocks'
            assert head in fblocks, 'fatal:invalid head or tail in freed blocks'
            prev_block = [head, tail]

        if end_pos in fblocks:
            head, tail = fblocks[end_pos]
            assert end_pos == head, 'fatal:invalid head pos in freed blocks'
            assert tail in fblocks, 'fatal:invalid head or tail in freed blocks'
            next_block = [head, tail]

        if not prev_block and not next_block:
            block_range = [start_pos, end_pos]
        elif prev_block and not next_block:
            block_range = [prev_block[0], end_pos]
            del fblocks[prev_block[0]]
            del fblocks[prev_block[1]]
        elif not prev_block and next_block:
            block_range = [start_pos, next_block[1]]
            del fblocks[next_block[0]]
            del fblocks[next_block[1]]
        elif prev_block and next_block:
            block_range = [prev_block[0], next_block[1]]
            del fblocks[prev_block[0]]
            del fblocks[prev_block[1]]
            del fblocks[next_block[0]]
            del fblocks[next_block[1]]

        assert block_range[0] not in fblocks and \
            block_range[1] not in fblocks, 'invalid block range[%s]' \
            % (str(block_range))

        fblocks[block_range[0]] = block_range
        fblocks[block_range[1]] = block_range

        self._statis['free_times'] += 1
        self._statis['used'] -= size
        return

    def put_data(self, shared_buf, data):
        """  put 'data' to 'shared_buf'
        """
        assert shared_buf.capacity() >= len(data), 'too large data[%d] '\
            'for this buffer[%s]' % (len(data), str(shared_buf))
        ctypes.memmove(self._base + shared_buf._pos, data, len(data))

    def get_data(self, shared_buf, start, size):
        """ get 'data' from this 'shared_buf'
        """
        return self._shared_mem[start:start + size]

    def get_statis(self):
        """ get statis about this memory manager
        """
        return str(self._statis)

    def __str__(self):
        return 'SharedMemoryMgr:%s' % (self.get_statis())

    def __del__(self):
        if self.s_log_statis:
            logger.info('destroy [%s]' % (self))
