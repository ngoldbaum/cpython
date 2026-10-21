/* Test PEP 688 - Buffers */

#include "parts.h"


#include <stddef.h>                 // offsetof

typedef struct {
    PyObject_HEAD
    PyObject *obj;
    Py_ssize_t references;
} testBufObject;

#define testBufObject_CAST(op)  ((testBufObject *)(op))

static PyObject *
testbuf_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *obj = PyBytes_FromString("test");
    if (obj == NULL) {
        return NULL;
    }
    testBufObject *self = (testBufObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        Py_DECREF(obj);
        return NULL;
    }
    self->obj = obj;
    self->references = 0;
    return (PyObject *)self;
}

static int
testbuf_traverse(PyObject *op, visitproc visit, void *arg)
{
    testBufObject *self = testBufObject_CAST(op);
    Py_VISIT(self->obj);
    return 0;
}

static int
testbuf_clear(PyObject *op)
{
    testBufObject *self = testBufObject_CAST(op);
    Py_CLEAR(self->obj);
    return 0;
}

static void
testbuf_dealloc(PyObject *op)
{
    testBufObject *self = testBufObject_CAST(op);
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->obj);
    Py_TYPE(self)->tp_free(self);
}

static int
testbuf_getbuf(PyObject *op, Py_buffer *view, int flags)
{
    testBufObject *self = testBufObject_CAST(op);
    int buf = PyObject_GetBuffer(self->obj, view, flags);
    if (buf == 0) {
        Py_SETREF(view->obj, Py_NewRef(self));
        self->references++;
    }
    return buf;
}

static void
testbuf_releasebuf(PyObject *op, Py_buffer *Py_UNUSED(view))
{
    testBufObject *self = testBufObject_CAST(op);
    self->references--;
    assert(self->references >= 0);
}

static PyBufferProcs testbuf_as_buffer = {
    .bf_getbuffer = testbuf_getbuf,
    .bf_releasebuffer = testbuf_releasebuf,
};

static struct PyMemberDef testbuf_members[] = {
    {"references", Py_T_PYSSIZET, offsetof(testBufObject, references), Py_READONLY},
    {NULL},
};

static PyTypeObject testBufType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "testBufType",
    .tp_basicsize = sizeof(testBufObject),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_new = testbuf_new,
    .tp_dealloc = testbuf_dealloc,
    .tp_traverse = testbuf_traverse,
    .tp_clear = testbuf_clear,
    .tp_as_buffer = &testbuf_as_buffer,
    .tp_members = testbuf_members
};

/* Get the pointer from a buffer-supporting object as a PyLong.
 *
 * Used to test alignment properties. */
static PyObject *
buffer_pointer_as_int(PyObject *Py_UNUSED(module), PyObject *obj)
{
    PyObject *out;
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    out = PyLong_FromVoidPtr(view.buf);
    PyBuffer_Release(&view);
    return out;
}

static PyObject *
clear_memoryview(PyObject *Py_UNUSED(module), PyObject *obj)
{
    if (!PyMemoryView_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "memoryview expected");
        return NULL;
    }
    if (Py_TYPE(obj)->tp_clear(obj) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* A native exporter with immutable storage. bad_result deliberately violates
 * the successful-export contract, using only valid owned references. */
typedef struct {
    PyObject_HEAD
    int bad_result;
    PyObject *delegate;
    Py_ssize_t acquisitions;
    Py_ssize_t releases;
} StableBufferObject;

static int
stablebuf_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    static const char data[] = "stable";
    StableBufferObject *self = (StableBufferObject *)obj;
    int result;
    Py_BEGIN_CRITICAL_SECTION(obj);
    self->acquisitions++;
    if (self->bad_result == 5) {
        PyErr_SetString(PyExc_BufferConflictError, "test conflict");
        view->obj = NULL;
        result = -1;
    }
    else if (self->bad_result == 6) {
        PyObject *delegate = PyByteArray_FromStringAndSize("delegated", 9);
        if (delegate == NULL) {
            result = -1;
        }
        else {
            Py_XSETREF(self->delegate, delegate);
            result = PyObject_GetBuffer(delegate, view,
                                        flags & ~PyBUF_STABLE_READ);
            if (result == 0) {
                view->readonly = 1;
            }
        }
    }
    else {
        result = PyBuffer_FillInfo(view, obj, (void *)data,
                                   sizeof(data) - 1, 1, flags);
        if (result == 0) {
            switch (self->bad_result) {
                case 1:
                    view->readonly = 0;
                    break;
                case 2:
                    view->readonly = 2;
                    break;
                case 3:
                    Py_SETREF(view->obj, Py_NewRef(Py_None));
                    break;
                case 4:
                    Py_CLEAR(view->obj);
                    break;
            }
        }
    }
    Py_END_CRITICAL_SECTION();
    return result;
}

static int
stablebuf_alternate_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    return stablebuf_getbuffer(obj, view, flags);
}

static void
stablebuf_releasebuffer(PyObject *obj, Py_buffer *Py_UNUSED(view))
{
    Py_BEGIN_CRITICAL_SECTION(obj);
    ((StableBufferObject *)obj)->releases++;
    Py_END_CRITICAL_SECTION();
}

static void
stablebuf_dealloc(PyObject *obj)
{
    PyTypeObject *type = Py_TYPE(obj);
    Py_XDECREF(((StableBufferObject *)obj)->delegate);
    type->tp_free(obj);
    if (type->tp_flags & Py_TPFLAGS_HEAPTYPE) {
        Py_DECREF(type);
    }
}

static PyMemberDef stablebuf_members[] = {
    {"bad_result", Py_T_INT, offsetof(StableBufferObject, bad_result)},
    {"delegate", Py_T_OBJECT_EX, offsetof(StableBufferObject, delegate), Py_READONLY},
    {"acquisitions", Py_T_PYSSIZET, offsetof(StableBufferObject, acquisitions), Py_READONLY},
    {"releases", Py_T_PYSSIZET, offsetof(StableBufferObject, releases), Py_READONLY},
    {NULL},
};

static PyBufferProcs stablebuf_as_buffer = {
    .bf_getbuffer = stablebuf_getbuffer,
    .bf_releasebuffer = stablebuf_releasebuffer,
};

static PyTypeObject staticStableBufType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_testcapi.StaticStableBuffer",
    .tp_basicsize = sizeof(StableBufferObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = stablebuf_dealloc,
    .tp_as_buffer = &stablebuf_as_buffer,
    .tp_members = stablebuf_members,
};

static PyObject *
make_stable_buffer_type(PyObject *Py_UNUSED(module), PyObject *args,
                        PyObject *kwargs)
{
    PyObject *capabilities = Py_None;
    int immutable = 1, nested = 0, alternate = 0, inherit_buffer = 0;
    PyObject *base = (PyObject *)&PyBaseObject_Type;
    PyObject *metaclass = (PyObject *)&PyType_Type;
    static char *keywords[] = {
        "capabilities", "immutable", "base", "nested", "alternate",
        "metaclass", "inherit_buffer", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OpOppOp", keywords,
                                    &capabilities, &immutable, &base,
                                    &nested, &alternate, &metaclass,
                                    &inherit_buffer)) {
        return NULL;
    }
    uint64_t mask = 0;
    if (capabilities != Py_None) {
        mask = PyLong_AsUnsignedLongLong(capabilities);
        if (mask == (uint64_t)-1 && PyErr_Occurred()) {
            return NULL;
        }
    }
    PySlot declaration[] = {
        PySlot_UINT64(Py_tp_buffer_capabilities, mask),
        PySlot_END
    };
    PySlot buffer_slots[] = {
        PySlot_FUNC(Py_bf_getbuffer, alternate ? stablebuf_alternate_getbuffer
                                             : stablebuf_getbuffer),
        PySlot_FUNC(Py_bf_releasebuffer, stablebuf_releasebuffer),
        PySlot_END
    };
    PySlot slots[] = {
        PySlot_DATA(Py_tp_name, "_testcapi.StableBuffer"),
        PySlot_SIZE(Py_tp_basicsize, sizeof(StableBufferObject)),
        PySlot_UINT64(Py_tp_flags, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
                     (immutable ? Py_TPFLAGS_IMMUTABLETYPE : 0)),
        PySlot_DATA(Py_tp_base, base),
        PySlot_DATA(Py_tp_metaclass, metaclass),
        PySlot_FUNC(Py_tp_new, PyType_GenericNew),
        PySlot_FUNC(Py_tp_dealloc, stablebuf_dealloc),
        PySlot_STATIC_DATA(Py_tp_members, stablebuf_members),
        PySlot_DATA(Py_slot_subslots, buffer_slots),
        PySlot_DATA(Py_slot_subslots, declaration),
        PySlot_END
    };
    if (inherit_buffer) {
        buffer_slots[0].sl_id = Py_slot_end;
    }
    if (capabilities == Py_None) {
        declaration[0].sl_id = Py_slot_end;
    }
    else if (!nested) {
        slots[Py_ARRAY_LENGTH(slots) - 2] = declaration[0];
    }
    return (PyObject *)PyType_FromSlots(slots);
}

static int
stable_module_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    return PyBuffer_FillInfo(view, obj, "stable", 6, 1, flags);
}

static PyObject *
make_stable_buffer_module_type(PyObject *Py_UNUSED(module), PyObject *arg)
{
    int declare = PyObject_IsTrue(arg);
    if (declare < 0) {
        return NULL;
    }
    PySlot slots[] = {
        PySlot_DATA(Py_tp_name, "_testcapi.StableBufferModule"),
        PySlot_DATA(Py_tp_base, &PyModule_Type),
        PySlot_UINT64(Py_tp_flags, Py_TPFLAGS_IMMUTABLETYPE),
        PySlot_FUNC(Py_bf_getbuffer, stable_module_getbuffer),
        PySlot_UINT64(Py_tp_buffer_capabilities, PyBUF_STABLE_READ),
        PySlot_END
    };
    if (!declare) {
        slots[Py_ARRAY_LENGTH(slots) - 2].sl_id = Py_slot_end;
    }
    return PyType_FromSlots(slots);
}

static PyObject *
stable_buffer_type_from_legacy_slots(PyObject *Py_UNUSED(module), PyObject *arg)
{
    int from_spec = PyObject_IsTrue(arg);
    if (from_spec < 0) {
        return NULL;
    }
    PyType_Slot legacy_slots[] = {
        {Py_bf_getbuffer, stable_module_getbuffer},
        {Py_tp_buffer_capabilities, (void *)(uintptr_t)PyBUF_STABLE_READ},
        {0, NULL}
    };
    if (from_spec) {
        PyType_Spec spec = {
            .name = "_testcapi.LegacyStableBuffer",
            .flags = Py_TPFLAGS_IMMUTABLETYPE,
            .slots = legacy_slots,
        };
        return PyType_FromSpec(&spec);
    }
    return PyType_FromSlots((PySlot[]) {
        PySlot_DATA(Py_tp_name, "_testcapi.LegacyStableBuffer"),
        PySlot_UINT64(Py_tp_flags, Py_TPFLAGS_IMMUTABLETYPE),
        PySlot_DATA(Py_tp_slots, legacy_slots),
        PySlot_END
    });
}

static PyObject *
register_buffer_capabilities(PyObject *Py_UNUSED(module), PyObject *args)
{
    PyTypeObject *type;
    unsigned long long capabilities;
    if (!PyArg_ParseTuple(args, "O!K", &PyType_Type, &type, &capabilities)) {
        return NULL;
    }
    if (PyUnstable_Type_RegisterBufferCapabilities(type, capabilities) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
get_buffer_bytes(PyObject *Py_UNUSED(module), PyObject *args)
{
    PyObject *obj;
    int flags;
    if (!PyArg_ParseTuple(args, "Oi", &obj, &flags)) {
        return NULL;
    }
    Py_buffer view;
    memset(&view, 0xcc, sizeof(view));
    if (PyObject_GetBuffer(obj, &view, flags) < 0) {
        if (flags & PyBUF_STABLE_READ) {
            assert(view.obj == NULL);
        }
        return NULL;
    }
    PyObject *result = PyBytes_FromStringAndSize(view.buf, view.len);
    PyBuffer_Release(&view);
    return result;
}

static PyMethodDef test_methods[] = {
    {"buffer_pointer_as_int", buffer_pointer_as_int, METH_O},
    {"clear_memoryview", clear_memoryview, METH_O},
    {"make_stable_buffer_type", _PyCFunction_CAST(make_stable_buffer_type),
     METH_VARARGS | METH_KEYWORDS},
    {"make_stable_buffer_module_type", make_stable_buffer_module_type, METH_O},
    {"stable_buffer_type_from_legacy_slots", stable_buffer_type_from_legacy_slots,
     METH_O},
    {"register_buffer_capabilities", register_buffer_capabilities, METH_VARARGS},
    {"get_buffer_bytes", get_buffer_bytes, METH_VARARGS},
    {NULL},
};

int
_PyTestCapi_Init_Buffer(PyObject *m) {
    if (PyType_Ready(&staticStableBufType) < 0
        || PyUnstable_Type_RegisterBufferCapabilities(
            &staticStableBufType, PyBUF_STABLE_READ) < 0
        || PyModule_AddType(m, &staticStableBufType) < 0
        || PyModule_AddIntConstant(m, "PyBUF_STABLE_READ", PyBUF_STABLE_READ) < 0
        || PyModule_AddObjectRef(m, "BufferConflictError", PyExc_BufferConflictError) < 0)
    {
        return -1;
    }
    if (PyType_Ready(&testBufType) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(m, "testBuf", (PyObject *)&testBufType)) {
        return -1;
    }
    if (PyModule_AddFunctions(m, test_methods) < 0) {
        return -1;
    }

    return 0;
}
