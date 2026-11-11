"""Run a benchmark with OS-level socket denial; no network calls can succeed."""
import ctypes
import ctypes.util
import errno
import runpy
import sys

def main(entrypoint=None):
    library = ctypes.util.find_library('seccomp') if sys.platform.startswith('linux') else None
    if not library:
        raise SystemExit('Offline evaluation requires Linux and libseccomp. Cache model weights first.')
    lib = ctypes.CDLL(library, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7fff0000)
    if not context:
        raise RuntimeError('Could not initialize network restriction')
    for name in ('socket', 'connect', 'sendto', 'sendmsg', 'sendmmsg'):
        syscall = lib.seccomp_syscall_resolve_name(name.encode())
        if syscall >= 0 and lib.seccomp_rule_add(context, 0x50000 | errno.EPERM, syscall, 0) != 0:
            raise RuntimeError('Could not restrict network syscall ' + name)
    if lib.seccomp_load(context) != 0:
        raise RuntimeError('Could not enforce network restriction')
    lib.seccomp_release(context)
    import socket
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        print('Offline guard active: socket creation denied by OS', flush=True)
    else:
        raise RuntimeError('Network restriction not active')
    from pathlib import Path
    script = str(entrypoint or Path(__file__).with_name('evaluate_ocr.py'))
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name='__main__')


if __name__ == "__main__":
    main()
