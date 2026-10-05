"""Offline fake-DLL tests; no SDK import, hardware, GUI, or network access."""
import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name("03Auto_Snake_Scan_Camera_v3.py")


def load_stage():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"_handle_value", "XIMCStage"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    ns = {"ctypes": ctypes, "threading": threading, "time": time,
          "platform": SimpleNamespace(system=lambda: "Windows"),
          # Existing bytes support evidence hashing; WinDLL is always patched.
          "_default_ximc_dll_path": lambda: str(SOURCE),
          "DEVICE_X": b"fake-x", "DEVICE_Y": b"fake-y",
          "INVERT_X": True, "INVERT_Y": False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["XIMCStage"]


class FakeFunction:
    """Allow ctypes signature assignment while keeping fake calls observable."""
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class FakeDLL:
    def __init__(self, version="3.0.2", mode=9):
        self.version = version
        self.mode = mode
        self.engine_type = 3
        self.positions = {11: 0, 22: 0}
        self.enc_positions = {11: 123, 22: 456}
        self.commands = []
        self.stops = []
        self.closed = []
        self.opened = []
        self.move_result = 0
        self.move_behavior = "normal"
        self.flags = 0
        self.command_status = 0
        self.move_state = 0
        self.io_log = []
        self.read_error = None
        self.after_move = None
        self.status_hook = None
        self.struct_reads = 0
        for name in ("ximc_version", "open_device", "close_device", "command_movr",
                     "command_sstp", "get_position", "get_status", "get_engine_settings",
                     "get_entype_settings"):
            setattr(self, name, FakeFunction(getattr(self, "_" + name)))

    def _ximc_version(self, buffer):
        buffer.value = self.version.encode("ascii")

    def _open_device(self, name):
        self.opened.append(name)
        return 11 if name == b"fake-x" else 22

    def _close_device(self, handle_ptr):
        assert isinstance(handle_ptr._obj, ctypes.c_int)
        self.closed.append(handle_ptr._obj.value)
        handle_ptr._obj.value = -1
        return 0

    def _command_movr(self, dev, steps, microsteps):
        self.io_log.append(("move", dev))
        self.commands.append((dev, steps, microsteps))
        if self.move_result:
            return self.move_result
        divisor = 1 << (self.mode - 1)
        if self.move_behavior == "normal":
            self.positions[dev] += steps * divisor + microsteps
        elif self.move_behavior == "wrong_delta":
            self.positions[dev] += (steps + 1) * divisor + microsteps
        elif self.move_behavior == "never_stops":
            self.command_status = 0x80
        elif self.move_behavior == "command_error":
            self.command_status = 0x40
        if self.after_move:
            self.after_move()
        return 0

    def _command_sstp(self, dev):
        self.io_log.append(("stop", dev))
        self.stops.append(dev)
        return 0

    def _read_result(self, name):
        self.struct_reads += 1
        return -1 if self.read_error == name else 0

    def _get_position(self, dev, out):
        rc = self._read_result("get_position")
        if rc:
            return rc
        divisor = 1 << (self.mode - 1)
        total = self.positions[dev]
        whole = (abs(total) // divisor) * (-1 if total < 0 else 1)
        out._obj.Position = whole
        out._obj.uPosition = total - whole * divisor
        out._obj.EncPosition = self.enc_positions[dev]
        return 0

    def _get_status(self, dev, out):
        rc = self._read_result("get_status")
        if rc:
            return rc
        out._obj.MvCmdSts = self.command_status
        out._obj.MoveSts = self.move_state
        out._obj.Flags = self.flags
        if self.status_hook:
            self.status_hook()
        return 0

    def _get_engine_settings(self, dev, out):
        rc = self._read_result("get_engine_settings")
        out._obj.MicrostepMode = self.mode
        out._obj.StepsPerRev = 200
        return rc

    def _get_entype_settings(self, dev, out):
        rc = self._read_result("get_entype_settings")
        out._obj.EngineType = self.engine_type
        return rc


def connected_stage(dll=None):
    dll = dll or FakeDLL()
    stage = load_stage()()
    with patch.object(ctypes, "WinDLL", return_value=dll, create=True):
        stage.connect()
    return stage, dll
