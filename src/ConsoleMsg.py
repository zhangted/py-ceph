class ConsoleMsg(object):
  BOLD = '\033[1m'
  RED = '\033[91m'
  END = '\033[0m'

  @staticmethod
  def print_terminate():
    input("Press Enter to Exit...")

  @staticmethod
  def print_err_terminate(msg):
    print(f"[Error] => {msg}")
    ConsoleMsg.print_terminate()