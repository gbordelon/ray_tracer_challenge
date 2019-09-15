class GlobalConfig(object):
    def __init__(self, thread_count=4, timeout=30, divide_threshold=1, clamping=False):
        self.thread_count = thread_count
        self.timeout = timeout
        self.divide_threshold = divide_threshold
        self.clamping = clamping

    @classmethod
    def from_yaml(cls, obj):
        thread_count = 4
        timeout = 30
        divide_threshold = 1
        clamping = False

        if 'thread-count' in obj:
            thread_count = int(obj['thread-count'])
        if 'timeout' in obj:
            timeout = int(obj['timeout'])
        if 'divide-threshold' in obj:
            divide_threshold = int(obj['divide-threshold'])
        if 'clamping' in obj:
            clamping = obj['clamping']

        return cls(thread_count=thread_count,
                   timeout=timeout,
                   divide_threshold=divide_threshold,
                   clamping=clamping)