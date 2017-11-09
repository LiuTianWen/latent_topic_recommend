class KVTtem:
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def value(self):
        return tuple((self.k, self.v))

    def __gt__(self, item):
        if self.v >= item.v:
            return True
        return False

    def __str__(self):
        return '(' + str(self.k) + ':' + str(self.v) + ')'

    __repr__ = __str__


class ZHeap:
    def __init__(self, item=[], maxsize=5):
        # 初始化。item为数组
        self.items = item
        self.heapsize = len(self.items)
        self.maxsize = maxsize

    def LEFT(self, i):
        return 2 * i + 1

    def RIGHT(self, i):
        return 2 * i + 2

    def PARENT(self, i):
        return (i - 1) // 2

    def MIN_HEAPIFY(self, i):
        # 最小堆化：使以i为根的子树成为最小堆
        l = self.LEFT(i)
        r = self.RIGHT(i)
        if l < self.heapsize and self.items[l] < self.items[i]:
            smallest = l
        else:
            smallest = i

        if r < self.heapsize and self.items[r] < self.items[smallest]:
            smallest = r

        if smallest != i:
            self.items[i], self.items[smallest] = self.items[smallest], self.items[i]
            self.MIN_HEAPIFY(smallest)

    def INSERT(self, val):
        # 插入一个值val，并且调整使满足堆结构
        self.items.append(val)
        idx = len(self.items) - 1
        parIdx = self.PARENT(idx)
        while parIdx >= 0:
            if self.items[parIdx] > self.items[idx]:
                self.items[parIdx], self.items[idx] = self.items[idx], self.items[parIdx]
                idx = parIdx
                parIdx = self.PARENT(parIdx)
            else:
                break
        self.heapsize += 1
        if self.heapsize > self.maxsize:
            self.DELETE()

    def DELETE(self):
        last = len(self.items) - 1
        if last < 0:
            # 堆为空
            return None
        # else:
        self.items[0], self.items[last] = self.items[last], self.items[0]
        val = self.items.pop()
        self.heapsize -= 1
        self.MIN_HEAPIFY(0)
        return val

    def BUILD_MIN_HEAP(self):
        # 建立最小堆, O(nlog(n))
        i = self.PARENT(len(self.items) - 1)
        while i >= 0:
            self.MIN_HEAPIFY(i)
            i -= 1

    def SHOW(self):
        print(self.items)


class ZPriorityQ(ZHeap):
    def __init__(self, item=[], maxsize=5):
        ZHeap.__init__(self, item, maxsize)

    def enQ(self, val):
        ZHeap.INSERT(self, val)

    def deQ(self):
        val = ZHeap.DELETE(self)
        return val

# a = [KVTtem('a',1),KVTtem('b',3),KVTtem('c',6),KVTtem('d',5),KVTtem('x',16),KVTtem('r',3),KVTtem('f',4),KVTtem('s',3),KVTtem('u',9),]
# print(KVTtem('a',1))
# pq = ZPriorityQ()
# n = len(a)
# for i in range(n):
#     pq.enQ(a[i])
#     pq.SHOW()
#
# while pq.heapsize > 0:
#     pq.deQ()
#     pq.SHOW()
