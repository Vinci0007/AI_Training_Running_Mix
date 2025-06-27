
from pathlib import Path
import webview

rootDir = Path(__file__).parent.parent


class Navigate2Page:
    def __init__(self):
        self.name = "Navigate2Page"

    def sayHello(self, name):
        return "Hello, " + name + "!"
    
    def navigate2aiAgent(self, htmlName):
        templatePath = rootDir / "templates"
        url = str(templatePath / htmlName)
        # webview.http.routed(url)
        webview.windows[0].load_url(f"file://{url}")
        return True