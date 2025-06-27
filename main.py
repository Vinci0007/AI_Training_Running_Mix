
import webview.http
import webview.http
from py_scripts import train as llm_train
from py_scripts import Navigate2Page

import os
import webview
from enum import Enum
from pathlib import Path

rootDir = Path(__file__).parent
platform = os.name


class apiList(dict):
    alertApi = None
    confirmApi = None
    navigateApi = Navigate2Page()
    testApi = Navigate2Page()


def create_window():
    api = apiList()
    mainPageUrl = rootDir / "templates/test.html"
    webview.create_window(
        "MyAIMixRobotApp", 
        str(mainPageUrl), 
        js_api=api, 
        width=1280, 
        height=720,
        min_size=(1024, 360),
        confirm_close=True)
    # webview.create_window("", 'https://pywebview.flowrl.com/')

if platform == "nt":
    import win32gui
    import win32con

    def set_icon(hwnd, icon_path):
        ico = win32gui.LoadImage(0, icon_path, win32con.IMAGE_ICON, 
                                0, 0, win32con.LR_LOADFROMFILE)
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, ico)    


if __name__ == "__main__":
    iconPath = rootDir / "static/assets/logo.ico"
    webview.settings['ALLOW_DOWNLOADS'] = True
    chineseQuitNotice = {
        'global.quitConfirmation': u' 确认关闭？'
    }

    create_window()
    webview.start(localization=chineseQuitNotice, icon=str(iconPath), debug=True)

    windows = webview.windows
    if windows and platform=="nt":
        set_icon(windows[0].handle, str(iconPath))
    else:
        ...
    # llm_train()