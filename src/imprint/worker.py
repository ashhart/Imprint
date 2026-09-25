import contextlib
import multiprocessing
import os
import sys
import threading


class WorkerError(RuntimeError):
    pass


def watch_parent(parent_pid, stopped):
    while not stopped.wait(0.5):
        if os.getppid() != parent_pid:
            os._exit(0)


def worker_main(connection, settings, backend_factory, parent_pid):
    stopped = threading.Event()
    monitor = threading.Thread(
        target=watch_parent, args=(parent_pid, stopped), daemon=True
    )
    monitor.start()
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from .runtime import Runtime

        with contextlib.redirect_stdout(sys.stderr):
            runtime = Runtime(*settings, backend_factory=backend_factory)
        connection.send(("ready", None))
        while True:
            command, arguments = connection.recv()
            if command == "close":
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = getattr(runtime, command)(**arguments)
                    if command == "stream":
                        for event in result:
                            connection.send(("event", event))
                        connection.send(("result", None))
                    else:
                        connection.send(("result", result))
            except Exception as error:
                connection.send(("error", (type(error).__name__, str(error))))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send(("error", (type(error).__name__, str(error))))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        stopped.set()
        connection.close()


class Worker:
    def __init__(self, settings, backend_factory=None):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=worker_main,
            args=(child, settings, backend_factory, os.getpid()),
            daemon=True,
        )
        self.process.start()
        child.close()

    def ready(self):
        try:
            kind, _ = self.receive()
            if kind != "ready":
                raise WorkerError("Model worker failed to initialize")
        except BaseException:
            self.close()
            raise

    def receive(self):
        try:
            kind, value = self.connection.recv()
        except (EOFError, OSError) as error:
            raise WorkerError("Model worker exited unexpectedly") from error
        if kind == "error":
            name, message = value
            if name in {"ValueError", "StoreError", "RecipeError"}:
                raise ValueError(message)
            raise WorkerError(f"{name}: {message}")
        return kind, value

    def call(self, command, **arguments):
        self.connection.send((command, arguments))
        kind, value = self.receive()
        if kind != "result":
            raise WorkerError("Invalid worker response")
        return value

    def stream(self, request):
        self.connection.send(("stream", {"request": request}))
        complete = False
        try:
            while True:
                kind, value = self.receive()
                if kind == "result":
                    complete = True
                    break
                if kind != "event":
                    raise WorkerError("Invalid worker stream event")
                yield value
        except (ValueError, WorkerError):
            complete = True
            raise
        finally:
            if not complete:
                self.close()

    def close(self):
        try:
            if self.process.is_alive():
                self.connection.send(("close", {}))
                self.process.join(timeout=0.5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)
        except (BrokenPipeError, EOFError, OSError):
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        finally:
            self.connection.close()
            self.process.join(timeout=2)
