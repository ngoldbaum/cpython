// Need limited C API version 3.16 for stable buffers.
#define Py_LIMITED_API 0x03100000

#include "parts.h"


static int
stable_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    return PyBuffer_FillInfo(view, obj, "stable", 6, 1, flags);
}

static PyObject *
make_stable_buffer_type(PyObject *Py_UNUSED(module),
                        PyObject *Py_UNUSED(args))
{
    return PyType_FromSlots((PySlot[]) {
        PySlot_DATA(Py_tp_name, "_testlimitedcapi.StableBuffer"),
        PySlot_UINT64(Py_tp_flags, Py_TPFLAGS_IMMUTABLETYPE),
        PySlot_FUNC(Py_tp_new, PyType_GenericNew),
        PySlot_FUNC(Py_bf_getbuffer, stable_getbuffer),
        PySlot_UINT64(Py_tp_buffer_capabilities, PyBUF_STABLE_READ),
        PySlot_END,
    });
}

static PyObject *
get_stable_buffer_bytes(PyObject *Py_UNUSED(module), PyObject *obj)
{
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE | PyBUF_STABLE_READ) < 0) {
        return NULL;
    }
    PyObject *result = PyBytes_FromStringAndSize(view.buf, view.len);
    PyBuffer_Release(&view);
    return result;
}

static PyMethodDef test_methods[] = {
    {"make_stable_buffer_type", make_stable_buffer_type, METH_NOARGS},
    {"get_stable_buffer_bytes", get_stable_buffer_bytes, METH_O},
    {NULL},
};

int
_PyTestLimitedCAPI_Init_Buffer(PyObject *module)
{
    if (PyModule_AddObjectRef(module, "BufferConflictError",
                              PyExc_BufferConflictError) < 0) {
        return -1;
    }
    return PyModule_AddFunctions(module, test_methods);
}
