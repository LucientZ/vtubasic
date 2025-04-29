import sys
sys.path.insert(0, "./src")
from app import App


if __name__ == "__main__":
    app: App = App()
    app.run()