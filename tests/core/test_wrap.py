# -*- coding: utf-8 -*-
import unittest

from sigmaepsilon.core.wrapping import Wrapper, customwrapper, wrap, wrapper


class TestWrap(unittest.TestCase):

    def test_wrap(self):
        class Wrapped:

            def foo(self):
                print('foo in object to be wrapped')

            def boo(self):
                print('boo in object to be wrapped')

        @customwrapper(wrapkey='wrapkey', wraptype=Wrapped)
        class CustomWrapper:

            def boo(self):
                print('boo in custom wrapper')

        @wrapper
        class DefaultWrapper:

            def boo(self):
                print('boo in default wrapper')

        class DefaultWrapper2(Wrapper):

            def boo(self):
                print('boo in default wrapper')

        cw = CustomWrapper(wrapkey=Wrapped())
        cw.foo()
        cw.boo()

        dw = DefaultWrapper(wrap=Wrapped())
        dw.foo()
        dw.boo()

        dw2 = DefaultWrapper(wrap=Wrapped())
        dw2.foo()
        dw2.boo()

        w = wrap(Wrapped())
        w.foo()
        w.boo()
        
        w.wrapped
        w.wraps()
        w.wrapped_obj()

        obj = CustomWrapper('a', Wrapped())
        
        @customwrapper(wrapkey='wrap', wraptype=dict)
        class CustomWrapper:

            def boo(self):
                print('boo in custom wrapper')
                
        obj = CustomWrapper('a', dict(a=2))
        assert obj['a'] == 2
        
        obj = CustomWrapper(a=2)
        assert obj['a'] == 2
        
if __name__ == "__main__":
    
    unittest.main()
