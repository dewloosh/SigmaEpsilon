# -*- coding: utf-8 -*-
import unittest
from abc import abstractmethod

from sigmaepsilon.core.abc import ABC_Safe, ABC_Weak, ABC_Strong


class TestMeta(unittest.TestCase):

    def test_meta_weak(self):
        """
        Explicit notation of an abstract method in a base class is
        not necessary. Error only occurs at runtime.
        """
        
        class ABC_Parent_Weak(ABC_Weak):

            @abstractmethod
            def abcParentWeak(self):
                pass

        class ABC_Child_Weak(ABC_Parent_Weak):

            @abstractmethod
            def abcChildWeak(self):
                pass

        class ABC_GrandChild_Weak(ABC_Child_Weak):

            @abstractmethod
            def abcGrandChildWeak(self):
                pass

        class GrandChild_Weak(ABC_GrandChild_Weak):
            """
            Comment out any of these implementations and witness how
            error arises on the line of object creation, but not during
            compilation.
            """

            def abcParentWeak(self):
                pass

            def abcChildWeak(self):
                pass

            def abcGrandChildWeak(self):
                pass

        grandchild_Weak = GrandChild_Weak()
        
        has_error = False
        try:
            grandchild_Weak = ABC_GrandChild_Weak()
        except:
            has_error = True
        assert has_error

    def test_meta_strong(self):
        """
        The code below throws a TypeError if an abstract method is not
        included in the definition of child class.
        Error occurs at compilation time.
        """
        
        class ABC_Parent_Strong(ABC_Strong):

            @abstractmethod
            def abcParentStrong(self):
                pass

        class ABC_Child_Strong(ABC_Parent_Strong):

            @abstractmethod
            def abcParentStrong(self):
                pass

            @abstractmethod
            def abcChildStrong(self):
                pass

        class ABC_GrandChild_Strong(ABC_Child_Strong):

            @abstractmethod
            def abcParentStrong(self):
                pass

            @abstractmethod
            def abcChildStrong(self):
                pass

            @abstractmethod
            def abcGrandChildStrong(self):
                pass

        class GrandChild_Strong(ABC_GrandChild_Strong):

            def abcParentStrong(self):
                pass

            def abcChildStrong(self):
                pass

            def abcGrandChildStrong(self):
                pass
        
        has_error = False
        try:
            class GrandChild_Strong(ABC_GrandChild_Strong):
                
                def abcParentStrong(self):
                    pass

                def abcChildStrong(self):
                    pass
        except:
            has_error = True
        assert has_error
        
    def test_meta_safe(self):
        
        class ABC_Parent_Safe(ABC_Safe):

            def funcParentSafe(self):
                pass

        class ABC_Child_Safe(ABC_Parent_Safe):

            def funcChildSafe(self):
                pass
            
        has_error = False
        try:
            class ABC_GrandChild_Safe(ABC_Child_Safe):

                def funcParentSafe(self):
                    pass
        except:
            has_error = True
        assert has_error
        
        
if __name__ == "__main__":
    unittest.main()
