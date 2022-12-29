import sys

class ConsoleMsg(object):
  @staticmethod
  def print_terminate():
    input("Press Enter to Exit...")
    sys.exit(0)

  @staticmethod
  def print_err_terminate(msg):
    print(f"[Error] => {msg}")
    ConsoleMsg.print_terminate()