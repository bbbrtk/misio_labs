#!/usr/bin/env python3
import numpy as np
import time


class Run:
    def __init__(self, p0, p, steps):
        self.p0 = p0
        self.p = p
        self.steps = steps
        self.states = {}


    def _formula_n(self, pdo):
        return pdo + (1-pdo)*self.p


    def _subformula(self, p0, steps, state):
        if steps > 0:
            # if state has been calculated  
            if (p0, steps, state) in self.states.keys():
                return self.states[(p0, steps, state)]
            # otherwise
            else:
                suck_result = 0
                go_result = 0
                np0 = self._formula_n(p0)

                if state == 'CLEAN':
                    # init
                    n0 = self._formula_n(0)

                    # SUCK
                    dirty = self._subformula(np0, steps-1, 'DIRTY')
                    clean = self._subformula(np0, steps-1, 'CLEAN')
                    suck_result = 0 + (1-self.p)*clean + self.p*dirty

                    # GO
                    dirty = self._subformula(n0, steps-1, 'DIRTY')
                    clean = self._subformula(n0, steps-1, 'CLEAN')
                    go_result = -1 + np0*dirty + (1 - np0)*clean
                        
                    # RESULT
                    result = max(suck_result, go_result)

                elif state == 'DIRTY':
                    # init
                    n1 = self._formula_n(1)

                    # SUCK
                    dirty = self._subformula(np0, steps-1, 'DIRTY')
                    clean = self._subformula(np0, steps-1, 'CLEAN')
                    suck_result = 10 + (1-self.p)*clean + self.p*dirty

                    # GO
                    dirty = self._subformula(n1, steps-1, 'DIRTY')
                    clean = self._subformula(n1, steps-1, 'CLEAN')
                    go_result = -1 + np0*dirty + (1 - np0)*clean
                        
                    # RESULT
                    result = max(suck_result, go_result)
                
                self.states[(p0, steps, state)] = result
                return result 
        
        else:
            return 0


    def _formula_v(self):
        dirty = self._subformula(self.p0, self.steps, 'DIRTY')
        clean = self._subformula(self.p0, self.steps, 'CLEAN')
        return (self.p0*dirty) + ((1-self.p0)*clean)


    def print_answer(self):
        answer = self._formula_v()
        print("{:0.5f}".format(answer), flush=True)


if __name__ == "__main__":
    n = int(input())
    for _ in range(n):
        p0, p, steps = [x for x in input().split()]
        p0 = np.float64(p0)
        p = np.float64(p)
        steps = int(steps)
        
        run = Run(p0, p, steps)
        run.print_answer()



