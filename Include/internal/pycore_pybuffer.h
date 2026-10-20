#ifndef Py_INTERNAL_PYBUFFER_H
#define Py_INTERNAL_PYBUFFER_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


extern int _PyObject_GetStableBuffer(PyObject *obj, Py_buffer *view,
                                    int flags, getbufferproc callback);

// Exported for the _interpchannels module.
PyAPI_FUNC(int) _PyBuffer_ReleaseInInterpreter(
        PyInterpreterState *interp, Py_buffer *view);
PyAPI_FUNC(int) _PyBuffer_ReleaseInInterpreterAndRawFree(
        PyInterpreterState *interp, Py_buffer *view);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_PYBUFFER_H */
