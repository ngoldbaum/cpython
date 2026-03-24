#ifndef Py_CPYTHON_CRITICAL_SECTION_H
#  error "this header file must not be included directly"
#endif

PyAPI_FUNC(void)
PyCriticalSection_BeginMutex(PyCriticalSection *c, PyMutex *m);

PyAPI_FUNC(void)
PyCriticalSection2_BeginMutex(PyCriticalSection2 *c, PyMutex *m1, PyMutex *m2);

#ifndef Py_GIL_DISABLED

#define Py_BEGIN_CRITICAL_SECTION_MUTEX(mutex)    \
    {
#define Py_BEGIN_CRITICAL_SECTION2_MUTEX(m1, m2) {

#else

#define Py_BEGIN_CRITICAL_SECTION_MUTEX(mutex)                         \
    {                                                                   \
        PyCriticalSection _py_cs;                                       \
        PyCriticalSection_BeginMutex(&_py_cs, mutex)

# define Py_BEGIN_CRITICAL_SECTION2_MUTEX(m1, m2)                       \
    {                                                                   \
        PyCriticalSection2 _py_cs2;                                     \
        PyCriticalSection2_BeginMutex(&_py_cs2, m1, m2)

#endif
