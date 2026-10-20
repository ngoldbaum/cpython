import gc
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
        # Memoryview support is a separate change; an ordinary view must not
        # bypass declaration checks by sharing its existing managed buffer.
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


if __name__ == '__main__':
    unittest.main()
