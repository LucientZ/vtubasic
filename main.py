import sys
sys.path.insert(0, "./src")
from app import *


if __name__ == "__main__":
    app: App = App()
    app.run()