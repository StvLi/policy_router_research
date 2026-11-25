#!/usr/bin/env python3
"""用 CDP 打开交互页，收集 JS 异常并截图（本机自检用，不进 repo 的 public 资产）。

用法：python3 scripts/chrome_check.py <html> <out.png> [检查表达式...]
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    html = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    exprs = sys.argv[3:]
    port = free_port()
    prof = tempfile.mkdtemp(prefix="cdp-prof-")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    proc = subprocess.Popen(
        [chrome, "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={prof}", "--allow-file-access-from-files",
         "--autoplay-policy=no-user-gesture-required",
         "--window-size=1800,1000", "--hide-scrollbars",
         "--no-first-run", "--no-default-browser-check", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ws_url = None
    for _ in range(100):
        try:
            data = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
            pages = [t for t in data if t.get("type") == "page"]
            if pages:
                ws_url = pages[0]["webSocketDebuggerUrl"]
                break
        except Exception:
            pass
        time.sleep(0.1)
    if not ws_url:
        print("无法连接 Chrome CDP")
        proc.kill()
        return 1

    ws = websocket.create_connection(ws_url, timeout=30)
    mid = [0]

    def send(method, **params):
        mid[0] += 1
        ws.send(json.dumps({"id": mid[0], "method": method, "params": params}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == mid[0]:
                return msg

    errors, logs = [], []
    send("Runtime.enable")
    send("Log.enable")
    send("Page.enable")
    send("Page.navigate", url="file://" + html)
    t0 = time.time()
    while time.time() - t0 < 8.0:
        ws.settimeout(0.5)
        try:
            msg = json.loads(ws.recv())
        except Exception:
            continue
        m = msg.get("method")
        if m == "Runtime.exceptionThrown":
            d = msg["params"]["exceptionDetails"]
            errors.append(f"{d.get('text')} :: "
                          f"{(d.get('exception') or {}).get('description', '')[:300]}")
        elif m == "Log.entryAdded":
            e = msg["params"]["entry"]
            if e.get("level") in ("error", "warning"):
                logs.append(f"[{e['level']}] {e.get('text')} {e.get('url','')}")
        elif m == "Runtime.consoleAPICalled":
            if msg["params"]["type"] in ("error", "warning"):
                logs.append("[console] " + " ".join(
                    str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])[:300])
    ws.settimeout(30)

    def ev(expr):
        r = send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return (r.get("result", {}).get("result", {}) or {}).get("value")

    print("== 异常 ==")
    print("\n".join(errors) if errors else "(无)")
    if logs:
        print("== 日志 ==")
        print("\n".join(logs[:20]))
    for e in exprs:
        try:
            print(f"== {e} ==\n{json.dumps(ev(e), ensure_ascii=False)}")
        except Exception as ex:
            print(f"== {e} == 求值失败: {ex}")

    shot = send("Page.captureScreenshot", format="png")
    data = shot.get("result", {}).get("data")
    if data:
        import base64
        open(out, "wb").write(base64.b64decode(data))
        print("截图:", out)
    ws.close()
    proc.kill()
    try:
        shutil.rmtree(prof, ignore_errors=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
