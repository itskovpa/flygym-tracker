"""Schedule cyclic collection on the GUI thread, between Qt event callbacks.

On this Windows/Python 3.14/PySide build, worker-triggered collection collided
with vial painting and caused native access violations. Guarding only the track
overlay missed other Qt drawing. Disable automatic collection before workers
start and collect from a Qt timer instead. Reference counting is unaffected;
cycles still get collected, including a full collection every 30 seconds.
"""
import gc
from PySide6.QtCore import QObject, QThread, QTimer, Slot


class GuiCollector(QObject):
    def __init__(self, app):
        super().__init__(app)
        if QThread.currentThread() != app.thread():
            raise RuntimeError('Install the collector on the GUI thread')
        self._was_enabled = gc.isenabled()
        self._stopped = False
        self._ticks = 0
        gc.disable()
        from flygym_tracker import diagnostics
        diagnostics.write('Cyclic garbage collection scheduled on the GUI thread (1 s; full sweep 30 s).')
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.collect)
        self.timer.start()
        app.aboutToQuit.connect(self.stop)

    @Slot()
    def collect(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError('Cyclic collection must run on the GUI thread')
        if self._stopped:
            return
        self._ticks += 1
        gc.collect(2 if self._ticks % 30 == 0 else 0)

    @Slot()
    def stop(self):
        if self._stopped:
            return
        self.timer.stop()
        self._stopped = True
        if self._was_enabled:
            gc.enable()


def install_gui_collector(app):
    collector = getattr(app, '_flygym_collector', None)
    if collector is None or collector._stopped:
        collector = GuiCollector(app)
        app._flygym_collector = collector
    return collector
