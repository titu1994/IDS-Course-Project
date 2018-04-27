import sys
sys.path.insert(0, "..")

from staging.servers.sentiment import app as application

if __name__ == '__main__':
    application.run()