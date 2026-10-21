import gc
import io
import pickle
import unittest
import weakref

from inspect import BufferFlags as F
from test import support
from test.support import import_helper, threading_helper


_testcapi = import_helper.import_module('_testcapi')
_testlimitedcapi = import_helper.import_module('_testlimitedcapi')

make_type = _testcapi.make_stable_buffer_type
register = _testcapi.register_buffer_capabilities
acquire = _testcapi.get_buffer_bytes
STABLE = F.STABLE_READ


class StableBufferTests(unittest.TestCase):
    def test_public_constants(self):
        self.assertEqual(STABLE, 0x400)
        self.assertEqual(STABLE, _testcapi.PyBUF_STABLE_READ)
        self.assertIs(_testcapi.BufferConflictError, BufferConflictError)
        self.assertIs(BufferConflictError.__base__, BufferError)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            exc = pickle.loads(pickle.dumps(BufferConflictError('busy'), protocol))
            self.assertIs(type(exc), BufferConflictError)
            self.assertEqual(exc.args, ('busy',))

    def test_limited_api_exporter(self):
        cls = _testlimitedcapi.make_stable_buffer_type()
        obj = cls()
        self.assertEqual(acquire(obj, STABLE), b'stable')
        self.assertEqual(_testlimitedcapi.get_stable_buffer_bytes(obj), b'stable')

    def test_limited_api_consumer(self):
        self.assertIs(_testlimitedcapi.BufferConflictError, BufferConflictError)
        obj = make_type(STABLE)()
        self.assertEqual(_testlimitedcapi.get_stable_buffer_bytes(obj), b'stable')
        self.assertEqual((obj.acquisitions, obj.releases), (1, 1))
        obj.bad_result = 5
        with self.assertRaises(BufferConflictError):
            _testlimitedcapi.get_stable_buffer_bytes(obj)
        self.assertEqual((obj.acquisitions, obj.releases), (2, 1))
        with self.assertRaises(BufferError):
            _testlimitedcapi.get_stable_buffer_bytes(make_type()())

    def test_static_declaration(self):
        cls = _testcapi.StaticStableBuffer
        register(cls, STABLE)
        obj = cls()
        self.assertEqual(acquire(obj, STABLE), b'stable')
        self.assertEqual((obj.acquisitions, obj.releases), (1, 1))

    def test_heap_declaration(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                cls = make_type(STABLE, nested=nested)
                self.assertEqual(acquire(cls(), STABLE), b'stable')
                register(cls, STABLE)  # Idempotent after a slot declaration.
                self.assertEqual(acquire(cls(), STABLE), b'stable')

    def test_inherited_buffer_slots(self):
        for capabilities in (None, STABLE):
            with self.subTest(capabilities=capabilities):
                base = make_type(capabilities)
                cls = make_type(STABLE, base=base, inherit_buffer=True)
                obj = cls()
                self.assertEqual(acquire(obj, STABLE), b'stable')
                self.assertEqual((obj.acquisitions, obj.releases), (1, 1))

    def test_registration(self):
        cls = make_type(immutable=False)
        with self.assertRaises(TypeError):
            register(cls, STABLE)
        _testlimitedcapi.type_freeze(cls)
        register(cls, STABLE)
        register(cls, STABLE)
        self.assertEqual(acquire(cls(), STABLE), b'stable')

    def test_invalid_capabilities(self):
        cls = make_type(STABLE)
        for mask in (0, 1, STABLE | F.WRITABLE, 1 << 32, (1 << 64) - 1):
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    register(cls, mask)
                with self.assertRaises(ValueError):
                    make_type(mask)
        self.assertEqual(acquire(cls(), STABLE), b'stable')

    def test_ineligible_types(self):
        with self.assertRaises(TypeError):
            make_type(STABLE, immutable=False)
        with self.assertRaises(TypeError):
            make_type(STABLE, inherit_buffer=True)
        with self.assertRaises(TypeError):
            register(object, STABLE)

        class PythonBuffer:
            def __buffer__(self, flags):
                raise AssertionError('callback must not run')

        _testlimitedcapi.type_freeze(PythonBuffer)
        with self.assertRaises(TypeError):
            register(PythonBuffer, STABLE)
        with self.assertRaises(BufferError):
            acquire(PythonBuffer(), STABLE)

        class Meta(type):
            pass

        with self.assertRaises(TypeError):
            make_type(STABLE, metaclass=Meta)
        base = make_type(immutable=False)
        with self.assertRaises(TypeError):
            make_type(STABLE, base=base)

    def test_module_types_rejected(self):
        make_module_type = _testcapi.make_stable_buffer_module_type
        with self.assertRaisesRegex(TypeError, 'module types'):
            make_module_type(True)
        cls = make_module_type(False)
        obj = cls('exporter')
        self.assertEqual(acquire(obj, F.SIMPLE), b'stable')
        with self.assertRaisesRegex(TypeError, 'module types'):
            register(cls, STABLE)
        with self.assertRaises(BufferError):
            acquire(obj, STABLE)

    def test_legacy_slots_rejected(self):
        # Reject both a PyType_Spec and PyType_Slot arrays nested in PySlot.
        for from_spec in (False, True):
            with self.subTest(from_spec=from_spec):
                with self.assertRaisesRegex(SystemError,
                                            'Py_tp_buffer_capabilities'):
                    _testcapi.stable_buffer_type_from_legacy_slots(from_spec)

    def test_not_inherited(self):
        base = make_type(STABLE)
        cls = make_type(base=base)
        obj = cls()
        with self.assertRaises(BufferError):
            acquire(obj, STABLE)
        self.assertEqual(obj.acquisitions, 0)
        declared = make_type(STABLE, base=base)
        self.assertEqual(acquire(declared(), STABLE), b'stable')

    def test_undeclared_exporter(self):
        obj = make_type()()
        with self.assertRaisesRegex(BufferError, 'has not declared') as cm:
            acquire(obj, STABLE)
        self.assertIs(type(cm.exception), BufferError)
        self.assertEqual(obj.acquisitions, 0)
        self.assertEqual(acquire(obj, F.SIMPLE), b'stable')

    def test_flags(self):
        obj = make_type(STABLE)()
        bases = (F.ND, F.STRIDES, F.C_CONTIGUOUS, F.F_CONTIGUOUS,
                 F.ANY_CONTIGUOUS, F.INDIRECT)
        valid = {STABLE}
        valid.update(STABLE | base for base in bases)
        valid.update(STABLE | base | F.FORMAT for base in bases)
        self.assertEqual(len(valid), 13)
        for flags in range(0x400, 0x800):
            with self.subTest(flags=flags):
                if flags in valid:
                    self.assertEqual(acquire(obj, flags), b'stable')
                else:
                    with self.assertRaises(ValueError):
                        acquire(obj, flags)
        for flags in (-1, -2, STABLE | 0x800, STABLE | (1 << 30)):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                acquire(obj, flags)
        self.assertEqual((obj.acquisitions, obj.releases), (13, 13))
        self.assertEqual(acquire(obj, STABLE | F.CONTIG_RO), b'stable')
        self.assertEqual(acquire(obj, STABLE | F.FULL_RO), b'stable')

    def test_error_order(self):
        for obj in (None, object(), b'bytes', bytearray(b'bytes')):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                acquire(obj, STABLE | F.WRITABLE)
        with self.assertRaises(TypeError):
            acquire(object(), STABLE)

    def test_malformed_success(self):
        cls = make_type(STABLE)
        for mode in range(1, 5):
            obj = cls()
            obj.bad_result = mode
            with self.subTest(mode=mode), self.assertRaises(SystemError):
                acquire(obj, STABLE)
            # Mode 3 returns None as owner, so the original exporter's
            # release callback must not run.
            releases = 0 if mode == 3 else 1
            self.assertEqual((obj.acquisitions, obj.releases), (1, releases))
            obj.bad_result = 0
            self.assertEqual(acquire(obj, STABLE), b'stable')
            self.assertEqual((obj.acquisitions, obj.releases), (2, releases + 1))

    def test_delegated_export_cleanup(self):
        for cls in (make_type(STABLE), _testcapi.StaticStableBuffer):
            for wrapper in (False, True):
                with self.subTest(cls=cls, wrapper=wrapper):
                    obj = cls()
                    obj.bad_result = 6
                    with self.assertRaises(SystemError):
                        if wrapper:
                            obj.__buffer__(STABLE)
                        else:
                            acquire(obj, STABLE)
                    self.assertEqual((obj.acquisitions, obj.releases), (1, 0))
                    # The returned bytearray's export must have been released.
                    obj.delegate.extend(b'!')
                    self.assertEqual(obj.delegate, b'delegated!')

    def test_callback_failure(self):
        obj = make_type(STABLE)()
        obj.bad_result = 5
        with self.assertRaises(BufferConflictError):
            acquire(obj, STABLE)
        self.assertEqual((obj.acquisitions, obj.releases), (1, 0))

    def test_buffer_wrapper(self):
        obj = make_type(STABLE)()
        with obj.__buffer__(STABLE | F.FULL_RO) as view:
            self.assertEqual(view.tobytes(), b'stable')
            self.assertTrue(view.readonly)
            self.assertEqual((obj.acquisitions, obj.releases), (1, 0))
        self.assertEqual(obj.releases, 1)
        undeclared = make_type()()
        with self.assertRaises(BufferError):
            undeclared.__buffer__(STABLE)
        self.assertEqual(undeclared.acquisitions, 0)
        base = make_type(STABLE)
        obj = make_type(STABLE, base=base, alternate=True)()
        with self.assertRaisesRegex(BufferError, 'does not match'):
            base.__buffer__(obj, STABLE)
        self.assertEqual(obj.acquisitions, 0)

    def test_memoryview_fast_path_checked(self):
        # An ordinary view must not gain a stable guarantee by sharing its
        # existing managed buffer, even when its exporter supports stable reads.
        with memoryview(b'data') as view:
            with self.assertRaises(BufferError):
                memoryview._from_flags(view, STABLE | F.FULL_RO)
            with self.assertRaises(ValueError):
                memoryview._from_flags(view, STABLE | F.WRITABLE)

    @support.thread_unsafe('requires synchronous garbage collection')
    def test_registration_does_not_keep_type_alive(self):
        cls = make_type(STABLE)
        ref = weakref.ref(cls)
        del cls
        support.gc_collect()
        self.assertIsNone(ref())
        # Exercise allocations after registered type objects have been freed.
        for _ in range(100):
            cls = make_type()
            with self.assertRaises(BufferError):
                acquire(cls(), STABLE)
            del cls
            gc.collect()

    @support.thread_unsafe('uses shared type watcher state')
    def test_resurrected_type_keeps_declaration(self):
        cls = make_type(STABLE)
        watcher = _testcapi.add_type_watcher(0)
        try:
            _testcapi.watch_type(watcher, cls)
            del cls
            support.gc_collect()
            events = _testcapi.get_type_modified_events()
            self.assertEqual(len(events), 1)
            cls = events[0]
            _testcapi.unwatch_type(watcher, cls)
            register(cls, STABLE)
            self.assertEqual(acquire(cls(), STABLE), b'stable')
            events.clear()
        finally:
            _testcapi.clear_type_watcher(watcher)

    def test_subinterpreter_declarations(self):
        cls = make_type(STABLE)
        for _ in range(2):
            self.assertEqual(support.run_in_subinterp('''
import _testcapi as capi
for cls in (capi.StaticStableBuffer, capi.make_stable_buffer_type(0x400)):
    capi.register_buffer_capabilities(cls, 0x400)
    assert capi.get_buffer_bytes(cls(), 0x400) == b'stable'
'''), 0)
            self.assertEqual(acquire(cls(), STABLE), b'stable')

    @support.cpython_only
    @support.thread_unsafe('changes process-wide memory allocators')
    def test_registration_does_not_allocate(self):
        cls = make_type()
        # Fail every allocation during both initial and repeated registration.
        for _ in range(2):
            _testcapi.call_with_nomemory(0, 0, register, cls, STABLE)
            self.assertEqual(acquire(cls(), STABLE), b'stable')

    @threading_helper.requires_working_threading()
    def test_concurrent_registration_and_acquisition(self):
        cls = make_type()
        obj = cls()

        def declare_and_read():
            for _ in range(100):
                register(cls, STABLE)
                self.assertEqual(acquire(obj, STABLE), b'stable')

        threading_helper.run_concurrently(declare_and_read, nthreads=4)
        self.assertEqual((obj.acquisitions, obj.releases), (400, 400))


class StableBytesTests(unittest.TestCase):
    def test_acquisition(self):
        for data in (b'', b'a', b'st\0able', bytes(range(256)) * 32):
            for cached_hash in (False, True):
                if cached_hash:
                    hash(data)
                for flags in (STABLE, STABLE | F.CONTIG_RO,
                              STABLE | F.FULL_RO):
                    with self.subTest(data=data[:8], cached_hash=cached_hash,
                                      flags=flags):
                        self.assertEqual(acquire(data, flags), data)
                self.assertEqual(
                    _testlimitedcapi.get_stable_buffer_bytes(data), data)
                self.assertEqual(_testcapi.bytes_stable_read(data, True), data)

    def test_buffer_wrapper(self):
        data = bytes(range(256))
        with data.__buffer__(STABLE | F.FULL_RO) as view:
            self.assertIs(view.obj, data)
            self.assertTrue(view.readonly)
            self.assertEqual(view.format, 'B')
            self.assertEqual(view.shape, (len(data),))
            self.assertEqual(view.strides, (1,))
            self.assertEqual(view.tobytes(), data)

    def test_subclass_not_declared(self):
        class BytesSubclass(bytes):
            pass

        obj = BytesSubclass(b'data')
        self.assertEqual(acquire(obj, F.SIMPLE), b'data')
        with self.assertRaises(BufferError):
            acquire(obj, STABLE)
        with self.assertRaises(BufferError):
            obj.__buffer__(STABLE)

    def test_resize(self):
        original = b'st\0able'
        for size in (0, 1, len(original), len(original) + 4):
            for cached_hash in (False, True):
                with self.subTest(size=size, cached_hash=cached_hash):
                    result = _testcapi.bytes_stable_resize(size, cached_hash)
                    expected = original[:size] + b'x' * max(
                        0, size - len(original))
                    self.assertEqual(result, expected)

    def test_concat(self):
        for concat in (_testlimitedcapi.bytes_concat,
                       _testlimitedcapi.bytes_concatanddel):
            data = bytes(range(256))
            with data.__buffer__(STABLE) as view:
                del data
                self.assertEqual(concat(view.obj, b'!'), bytes(range(256)) + b'!')
                self.assertEqual(view.tobytes(), bytes(range(256)))

    def test_bytesio_shared_storage(self):
        original = bytes(range(256))
        for method in ('getvalue', 'read', 'peek'):
            for operation in ('write', 'grow', 'truncate', 'getbuffer',
                              'reinitialize', 'close'):
                with self.subTest(method=method, operation=operation):
                    stream = io.BytesIO()
                    stream.write(original)
                    stream.seek(0)
                    data = getattr(stream, method)()
                    with data.__buffer__(STABLE) as view:
                        del data
                        if operation == 'write':
                            stream.seek(0)
                            stream.write(b'changed')
                        elif operation == 'grow':
                            stream.seek(0, io.SEEK_END)
                            stream.write(b'changed')
                        elif operation == 'truncate':
                            stream.truncate(1)
                        elif operation == 'getbuffer':
                            with stream.getbuffer() as writable:
                                writable[0] = 42
                        elif operation == 'reinitialize':
                            stream.__init__(b'changed')
                        else:
                            stream.close()
                        self.assertEqual(view.tobytes(), original)
                    stream.close()

    def test_bytearray_take_bytes(self):
        for count in (1, 4, 8):
            with self.subTest(count=count):
                source = bytearray(b'abcdefgh')
                data = source.take_bytes(count)
                with data.__buffer__(STABLE) as view:
                    del data
                    source[:] = b'changed'
                    self.assertEqual(view.tobytes(), b'abcdefgh'[:count])

    def test_subinterpreters(self):
        for _ in range(2):
            self.assertEqual(support.run_in_subinterp('''
import _testcapi
assert _testcapi.get_buffer_bytes(b'data', 0x400) == b'data'
'''), 0)

    @threading_helper.requires_working_threading()
    def test_concurrent_bytesio_writes(self):
        original = bytes(range(256)) * 32
        with io.BytesIO() as stream:
            stream.write(original)
            data = stream.getvalue()
            with data.__buffer__(STABLE) as view:
                del data

                def read():
                    for _ in range(100):
                        # This helper reads the bytes with the GIL released.
                        self.assertEqual(
                            _testcapi.bytes_stable_read(view.obj), original)

                def write():
                    replacement = b'x' * len(original)
                    for _ in range(100):
                        stream.seek(0)
                        stream.write(replacement)

                threading_helper.run_concurrently([read, write])
                self.assertEqual(view.tobytes(), original)

    @threading_helper.requires_working_threading()
    def test_concurrent_reads(self):
        data = bytes(range(256)) * 32

        def read():
            for _ in range(100):
                self.assertEqual(acquire(data, STABLE), data)
                self.assertEqual(_testcapi.bytes_stable_read(data), data)

        threading_helper.run_concurrently(read, nthreads=4)


if __name__ == '__main__':
    unittest.main()
class StableMemoryviewTests(unittest.TestCase):
    def test_constructor(self):
        for data in (b'', b'a', bytes(range(256))):
            with self.subTest(data=data):
                with memoryview(data, access='stable-read') as view:
                    self.assertIs(view.obj, data)
                    self.assertTrue(view.readonly)
                    self.assertEqual(view.tobytes(), data)
                    self.assertEqual(acquire(view, STABLE), data)
                    self.assertEqual(_testcapi.buffer_pointer_as_int(view),
                                     _testcapi.buffer_pointer_as_int(data))
        with self.assertRaises(BufferError):
            memoryview(bytearray(b'data'), access='stable-read')
        with self.assertRaises(TypeError):
            memoryview(object(), access='stable-read')

    def test_access_argument(self):
        for value in ('', 'read', 'STABLE-READ', 'stable-read\0'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                memoryview(b'data', access=value)
        for value in (False, 1, b'stable-read', object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                memoryview(b'data', access=value)
        with self.assertRaises(TypeError):
            memoryview(b'data', 'stable-read')
        with memoryview(object=b'data', access=None) as view:
            with self.assertRaises(BufferError):
                acquire(view, STABLE)
        class Access(str):
            def __eq__(self, other):
                raise AssertionError('must not call Python comparison')
        with memoryview(b'data', access=Access('stable-read')) as view:
            self.assertEqual(acquire(view, STABLE), b'data')

    def test_no_upgrade(self):
        for data in (b'data', bytearray(b'data')):
            with memoryview(data) as ordinary:
                views = [ordinary, ordinary[:], ordinary.cast('B'),
                         ordinary.toreadonly(), memoryview(ordinary)]
                for view in views:
                    with view, self.subTest(data=type(data), view=view):
                        with self.assertRaises(BufferError) as cm:
                            memoryview(view, access='stable-read')
                        self.assertIs(type(cm.exception), BufferError)
                        with self.assertRaises(BufferError):
                            acquire(view, STABLE)
                        with self.assertRaises(BufferError):
                            view.__buffer__(STABLE | F.FULL_RO)

    def test_shared_acquisition(self):
        for derive in (memoryview,
                       lambda v: memoryview(v, access='stable-read'),
                       lambda v: v[1:], lambda v: v.cast('H'),
                       lambda v: v.cast('B', shape=(2, 3)),
                       lambda v: v.toreadonly()):
            with self.subTest(derive=derive):
                obj = make_type(STABLE)()
                source = memoryview(obj, access='stable-read')
                child = derive(source)
                expected = child.tobytes()
                self.assertIs(child.obj, obj)
                self.assertTrue(child.readonly)
                source.release()
                self.assertEqual((obj.acquisitions, obj.releases), (1, 0))
                self.assertEqual(acquire(child, STABLE | F.FULL_RO), expected)
                self.assertEqual(acquire(child, F.FULL_RO), expected)
                self.assertEqual(_testlimitedcapi.get_stable_buffer_bytes(child),
                                 expected)
                with self.assertRaises(TypeError):
                    child[0] = 0
                for flags in (F.WRITABLE, F.FULL):
                    with self.assertRaises(BufferError):
                        acquire(child, flags)
                    with self.assertRaises(BufferError):
                        memoryview._from_flags(child, flags)
                with self.assertRaises(ValueError):
                    acquire(child, STABLE | F.WRITABLE)
                child.release()
                self.assertEqual((obj.acquisitions, obj.releases), (1, 1))
                with self.assertRaises(ValueError):
                    memoryview(child, access='stable-read')
                with self.assertRaises(ValueError):
                    acquire(child, STABLE)

    def test_buffer_wrapper_provenance(self):
        obj = make_type(STABLE)()
        source = obj.__buffer__(STABLE | F.FULL_RO)
        self.assertIs(source.obj, obj)
        for flags in (F.FULL_RO, STABLE | F.FULL_RO):
            with source.__buffer__(flags) as exported:
                # The __buffer__ wrapper owns a C re-export from source.
                self.assertIs(exported.obj, source)
                self.assertEqual(acquire(exported, STABLE), b'stable')
                with self.assertRaises(BufferError):
                    source.release()
                self.assertEqual(obj.releases, 0)
        source.release()
        self.assertEqual((obj.acquisitions, obj.releases), (1, 1))

    def test_layout(self):
        from _testbuffer import ndarray

        with memoryview(b'abcdef', access='stable-read') as source:
            views = [source[::2], source[::-1],
                     source.cast('B', shape=(2, 3)),
                     source.cast('B', shape=(2, 3), order='F'),
                     source[:1].cast('B', shape=())]
            source.release()
            for view in views:
                with view, self.subTest(shape=view.shape, strides=view.strides):
                    export = ndarray(view, getbuf=STABLE | F.FULL_RO)
                    self.assertEqual(export.tobytes(), view.tobytes())
                    self.assertEqual(export.shape, view.shape)
                    self.assertEqual(export.strides, view.strides)
                    self.assertEqual(export.format, view.format)
                    self.assertEqual(export.itemsize, view.itemsize)
                    self.assertEqual(export.suboffsets, view.suboffsets)
                    with self.assertRaises(BufferError):
                        view.release()
                    del export
                    for base, contiguous in ((F.C_CONTIGUOUS, view.c_contiguous),
                                             (F.F_CONTIGUOUS, view.f_contiguous),
                                             (F.ANY_CONTIGUOUS, view.contiguous)):
                        if contiguous:
                            export = ndarray(view, getbuf=STABLE | base | F.FORMAT)
                            self.assertEqual(export.tobytes(), view.tobytes())
                            del export
                        else:
                            with self.assertRaises(BufferError):
                                ndarray(view, getbuf=STABLE | base | F.FORMAT)

    def test_c_export_lifetime(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                obj = make_type(STABLE)()
                source = memoryview(obj, access='stable-read')
                child = source[:]
                def callback():
                    source.release()
                    with self.assertRaises(BufferError) as cm:
                        child.release()
                    self.assertIs(type(cm.exception), BufferError)
                    self.assertEqual(obj.releases, 0)
                    if fail:
                        raise RuntimeError('callback failed')
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'callback failed'):
                        _testcapi.stable_buffer_read(child, callback)
                else:
                    self.assertEqual(_testcapi.stable_buffer_read(child, callback),
                                     b'stable')
                self.assertEqual(obj.releases, 0)
                child.release()
                self.assertEqual((obj.acquisitions, obj.releases), (1, 1))

    def test_raw_wrappers_cannot_claim_stability(self):
        with memoryview(b'data', access='stable-read') as source:
            for from_buffer in (False, True):
                with _testcapi.memoryview_from_raw(source, from_buffer) as raw:
                    self.assertIsNone(raw.obj)
                    self.assertTrue(raw.readonly)
                    self.assertEqual(raw.tobytes(), b'data')
                    with self.assertRaises(BufferError):
                        acquire(raw, STABLE)
                    with self.assertRaises(BufferError):
                        memoryview(raw, access='stable-read')

    @support.thread_unsafe('requires synchronous garbage collection')
    def test_gc_lifetime(self):
        obj = make_type(STABLE)()
        source = memoryview(obj, access='stable-read')
        child = source[1:]
        cycle = [source]
        cycle.append(cycle)
        ref = weakref.ref(source)
        del source, cycle
        support.gc_collect()
        self.assertIsNone(ref())
        self.assertEqual(obj.releases, 0)
        self.assertEqual(acquire(child, STABLE), b'table')
        cycle = [child]
        cycle.append(cycle)
        del child, cycle
        support.gc_collect()
        self.assertEqual((obj.acquisitions, obj.releases), (1, 1))

    @threading_helper.requires_working_threading()
    def test_operation_lifetime(self):
        from threading import Event, Thread

        obj = make_type(STABLE)()
        view = memoryview(obj, access='stable-read')
        started = Event()
        resume = Event()
        releases = []
        class Other:
            def __buffer__(self, flags):
                started.set()
                if not resume.wait(support.SHORT_TIMEOUT):
                    raise RuntimeError('release thread did not finish')
                return memoryview(b'stable')
        def release():
            try:
                if started.wait(support.SHORT_TIMEOUT):
                    with self.assertRaises(BufferError) as cm:
                        view.release()
                    self.assertIs(type(cm.exception), BufferError)
                    releases.append(obj.releases)
            finally:
                resume.set()
        with threading_helper.start_threads([Thread(target=release)]):
            self.assertTrue(view == Other())
        self.assertEqual(releases, [0])
        self.assertEqual(view.tobytes(), b'stable')
        view.release()
        self.assertEqual((obj.acquisitions, obj.releases), (1, 1))

    @threading_helper.requires_working_threading()
    def test_concurrent_derivation_and_release(self):
        from threading import Barrier

        for derive in (lambda v: memoryview(v, access='stable-read'),
                       lambda v: v[1:], lambda v: v.cast('H'),
                       lambda v: v.toreadonly(),
                       lambda v: v.__buffer__(STABLE | F.FULL_RO)):
            for _ in range(20):
                obj = make_type(STABLE)()
                source = memoryview(obj, access='stable-read')
                barrier = Barrier(2)
                def read():
                    barrier.wait(support.SHORT_TIMEOUT)
                    try:
                        child = derive(source)
                    except ValueError:
                        return  # Release won the race.
                    with child:
                        expected = child.tobytes()
                        self.assertEqual(_testcapi.stable_buffer_read(child,
                            lambda: None), expected)
                def release():
                    barrier.wait(support.SHORT_TIMEOUT)
                    try:
                        source.release()
                    except BufferError:
                        pass  # Derivation still holds an operation pin.
                threading_helper.run_concurrently([read, release])
                source.release()
                self.assertEqual((obj.acquisitions, obj.releases), (1, 1))
