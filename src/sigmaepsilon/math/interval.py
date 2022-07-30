# -*- coding: utf-8 -*-
from numba.extending import unbox, NativeValue
from numba.extending import box
from numba.core import cgutils
from numba.extending import lower_builtin
from numba.extending import overload_attribute, overload
from numba.extending import make_attribute_wrapper
from numba.extending import models, register_model
from numba.extending import type_callable
from numba.extending import typeof_impl
from numba import types
import numpy as np
import operator


class Interval(object):
    """
    A half-open interval on the real number line.
    """

    def __init__(self, lo, hi, arr):
        self.lo = lo
        self.hi = hi
        self._arr = arr

    def __repr__(self):
        return 'Interval(%f, %f)' % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo

    def mean(self):
        return np.mean(self._arr)

    def __getitem__(self, key):
        return self._arr[key]


class IntervalType(types.Type):
    def __init__(self):
        self.data = types.Array(types.float64, 1, 'C')
        super(IntervalType, self).__init__(name='Interval')


make_attribute_wrapper(IntervalType, 'lo', 'lo')
make_attribute_wrapper(IntervalType, 'hi', 'hi')
make_attribute_wrapper(IntervalType, 'data', 'data')


@typeof_impl.register(Interval)
def typeof_index(val, c):
    return IntervalType()


@type_callable(Interval)
def type_interval(context):
    def typer(lo, hi, data):
        if isinstance(lo, types.Float) and isinstance(hi, types.Float):
            return IntervalType()
    return typer


@register_model(IntervalType)
class StructModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('lo', types.float64),
            ('hi', types.float64),
            ('data', fe_type.data),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@overload_attribute(IntervalType, "width")
def get_width(interval):
    def getter(interval):
        return interval.hi - interval.lo
    return getter


@overload_attribute(IntervalType, "mean")
def get_mean(interval):
    def getter(interval):
        return np.mean(interval.data)
    return getter


@overload(operator.getitem)
def overload_dummy_getitem(obj, idx):
    if isinstance(obj, IntervalType):

        def dummy_getitem_impl(obj, idx):
            return obj.data[idx]

        return dummy_getitem_impl


@lower_builtin(Interval, types.Float, types.Float, types.Array(types.float64, 1, 'C'))
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    lo, hi, data = args
    interval = cgutils.create_struct_proxy(typ)(context, builder)
    interval.lo = lo
    interval.hi = hi
    interval.data = data
    return interval._getvalue()


@unbox(IntervalType)
def unbox_interval(typ, obj, c):
    """
    Convert a Interval object to a native interval structure.
    """
    lo_obj = c.pyapi.object_getattr_string(obj, "lo")
    hi_obj = c.pyapi.object_getattr_string(obj, "hi")
    data_obj = c.pyapi.object_getattr_string(obj, "_arr")
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    interval.lo = c.pyapi.float_as_double(lo_obj)
    interval.hi = c.pyapi.float_as_double(hi_obj)
    interval.data = c.unbox(typ.data, data_obj).value
    c.pyapi.decref(lo_obj)
    c.pyapi.decref(hi_obj)
    c.pyapi.decref(data_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(interval._getvalue(), is_error=is_error)


@box(IntervalType)
def box_interval(typ, val, c):
    """
    Convert a native interval structure to an Interval object.
    """
    interval = cgutils.create_struct_proxy(
        typ)(c.context, c.builder, value=val)
    lo_obj = c.pyapi.float_from_double(interval.lo)
    hi_obj = c.pyapi.float_from_double(interval.hi)
    data_obj = c.box(typ.data, interval.data)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Interval))
    res = c.pyapi.call_function_objargs(class_obj, (lo_obj, hi_obj, data_obj))
    c.pyapi.decref(lo_obj)
    c.pyapi.decref(hi_obj)
    c.pyapi.decref(data_obj)
    c.pyapi.decref(class_obj)
    return res