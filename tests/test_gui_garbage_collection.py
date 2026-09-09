import gc
import threading
import weakref
from flygym_tracker.gui.garbage_collection import install_gui_collector


def test_cycles_are_collected_on_gui_thread_after_worker_allocation(qapp, pump):
    collector = install_gui_collector(qapp)
    events = []
    refs = []
    main = threading.get_ident()

    class Cycle:
        pass

    def allocate():
        for _ in range(10000):
            obj = Cycle()
            obj.self = obj
            refs.append(weakref.ref(obj))

    def callback(phase, info):
        if phase == 'start':
            events.append(threading.get_ident())

    gc.callbacks.append(callback)
    try:
        assert install_gui_collector(qapp) is collector
        assert not gc.isenabled()
        worker = threading.Thread(target=allocate)
        worker.start()
        worker.join()
        assert events == []
        assert any(ref() is not None for ref in refs)
        collector.timer.setInterval(10)
        pump(lambda: bool(events))
        assert events and set(events) == {main}
        collector._ticks = 29
        collector.collect()
        assert all(ref() is None for ref in refs)
    finally:
        gc.callbacks.remove(callback)
        collector.stop()
    assert gc.isenabled()


def test_existing_disabled_gc_is_not_enabled_on_shutdown(qapp):
    gc.disable()
    try:
        collector = install_gui_collector(qapp)
        collector.stop()
        collector.stop()
        assert not gc.isenabled()
    finally:
        gc.enable()
