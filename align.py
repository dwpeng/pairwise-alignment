from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from typing import Mapping

import numpy as np


class Path(IntEnum):
    F = 1
    H = 2
    E = 3


class AlignmentBase(ABC):

    def __init__(
        self,
        target: str,
        query: str,
        M: int,
        X: int,
        E: int
    ):
        self.target = target
        self.query = query
        self.M = M
        self.X = X
        self.E = E

        self.target_len = len(target)
        self.query_len = len(query)

        # 打分矩阵
        self.score_matrix = np.zeros(
            (self.target_len + 1, self.query_len + 1),
            dtype=np.int32
        )

        # 初始化打分矩阵
        for i in range(self.target_len+1):
            self.score_matrix[i, 0] = i * E
        for i in range(self.query_len+1):
            self.score_matrix[0, i] = i * E

        # 回溯路径
        self.traceback = np.zeros_like(
            self.score_matrix,
            dtype=np.int8
        )

        # 比对路径
        self.path = []

        # 最高分数
        self.max_score = -10000000

        # 最高分数出现的位置
        self.max_t, self.max_q = self.target_len, self.query_len

    def align(self, min_s=-1000000):
        """ 比对 """
        for tindex, t in enumerate(self.target, start=1):
            for qindex, q in enumerate(self.query, start=1):
                s = self.M if t == q else self.X
                f = self.score_matrix[tindex, qindex-1] + self.E
                e = self.score_matrix[tindex-1, qindex] + self.E
                h = s + self.score_matrix[tindex-1, qindex-1]
                s = max(h, f, e, min_s)
                # 局部比对需要从最高值开始回溯
                if s >= self.max_score:
                    self.max_score = s
                    self.max_t, self.max_q = tindex, qindex

                self.score_matrix[tindex, qindex] = s
                if h >= f and h >= e:
                    b = Path.H
                elif f >= e:
                    b = Path.F
                else:
                    b = Path.E
                self.traceback[tindex, qindex] = b
        return self

    @abstractmethod
    def do_traceback(self):
        raise NotImplementedError

    def _update(
        self,
        path_flag,
        tlen,
        qlen,
    ):
        """ 更新回溯路径 """
        update_path_map = {
            Path.H: (-1, -1),
            Path.F: (0, -1),
            Path.E: (-1, 0)
        }
        if path_flag == Path.H:
            self.path.append((
                (tlen-1, self.target[tlen-1]),
                (qlen-1, self.query[qlen-1])
            ))
        elif path_flag == Path.E:
            self.path.append((
                (tlen-1, self.target[tlen-1]),
                (None, None)
            ))
        elif path_flag == Path.F:
            self.path.append((
                (None, None),
                (qlen-1, self.query[qlen-1]),
            ))

        return update_path_map[path_flag]

    def print(self):
        print_path = []
        for (_, t), (_, q) in self.path:
            if t is None or q is None:
                format = '-'
            elif t == q:
                format = '|'
            else:
                format = '*'

            if t == None:
                t = ' '
            if q == None:
                q = ' '
            print_path.append((t, q, format))
        print(''.join([i[0] for i in print_path]))
        print(''.join([i[2] for i in print_path]))
        print(''.join([i[1] for i in print_path]))


class AlignmentGlobal(AlignmentBase):

    def do_traceback(self):
        """ 回溯路径 """
        tlen, qlen = self.target_len, self.query_len
        while tlen > 0 and qlen > 0:
            p = self.traceback[tlen, qlen]
            update_target, update_query = self._update(p, tlen, qlen)
            tlen, qlen = tlen + update_target, qlen + update_query

        if tlen > 0 or qlen > 0:
            if tlen > 0:
                for t in range(tlen, 0, -1):
                    self._update(Path.E, t, 1)
            else:
                for q in range(qlen, 0, -1):
                    self._update(Path.F, 1, q)

        self.path = self.path[::-1]
        return self


class AlignmentLocal(AlignmentBase):

    def align(self):
        return super().align(0)

    def do_traceback(self):
        start = self.score_matrix[self.max_t, self.max_q]
        # 局部比对到0结束
        while start != 0:
            p = self.traceback[self.max_t, self.max_q]
            update_t, update_q = self._update(p, self.max_t, self.max_q)
            self.max_t, self.max_q = self.max_t + update_t, self.max_q + update_q
            start = self.score_matrix[self.max_t, self.max_q]

        if self.max_t > 0 or self.max_q > 0:
            self._update(Path.H, self.max_t, self.max_q)

        self.path = self.path[::-1]
        return self


class AlignMode(Enum):
    GLOBAL = 'global'
    LOCAL = 'local'


def align(tseq, qseq, M, X, E, mode=AlignMode.GLOBAL) -> AlignmentBase:
    align_cls: Mapping[AlignMode, AlignmentBase] = {
        AlignMode.GLOBAL: AlignmentGlobal,
        AlignMode.LOCAL: AlignmentLocal
    }

    ali = align_cls[mode]
    return (
        ali(tseq, qseq, M, X, E)
        .align()
        .do_traceback()
    )


if __name__ == '__main__':
    tseq = 'GAGCT'
    qseq = 'GAGCTXXXX'
    M, X, E = 2, -4, -2
    # 全局比对
    a = align(tseq, qseq, M, X, E, mode=AlignMode.GLOBAL)
    a.print()

    # 局部比对
    a = align(tseq, qseq, M, X, E, mode=AlignMode.LOCAL)
    a.print()
