import sys
sys.path.insert(0, "..")

from staging.servers.ratings import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001, debug=False)