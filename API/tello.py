import socket
import threading
import time
from .stats import Stats


class Tello:
    def __init__(self, local_ip='', local_port=8889, tello_ip='192.168.10.1', tello_port=8889, max_timeout=15.0):
        self.local_ip = local_ip
        self.local_port = local_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket.bind((self.local_ip, self.local_port))

        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        self.tello_ip = tello_ip
        self.tello_port = tello_port
        self.tello_address = (self.tello_ip, self.tello_port)
        self.log = []

        self.max_timeout = max_timeout

    def send_command(self, command):
        """
        Send a command to the ip address. Will be blocked until
        the last command receives an 'OK'.
        If the command fails (either b/c time out or error),
        will try to resend the command
        :param command: (str) the command to send
        :param ip: (str) the ip of Tello
        :return: The latest command response
        """
        self.log.append(Stats(command, len(self.log)))

        self.socket.sendto(command.encode('utf-8'), self.tello_address)
        print(f'sending command: {command} to {self.tello_ip}')

        start = time.time()
        while not self.log[-1].got_response():
            now = time.time()
            diff = now - start
            if diff > self.max_timeout:
                print(f'Max timeout exceeded... command {command}')
                # TODO: is timeout considered failure or next command still get executed
                # now, next one got executed
                return
        print(f'Done!!! sent command: {command} to {self.tello_ip}')

    def _receive_thread(self):
        """Listen to responses from the Tello.

        Runs as a thread, sets self.response to whatever the Tello last returned.

        """
        while True:
            try:
                self.response, ip = self.socket.recvfrom(1024)
                print('from %s: %s' % (ip, self.response))

                self.log[-1].add_response(self.response)
            except socket.error as se:
                print(f"Caught exception socket.error : {se}")

    def on_close(self):
        pass
        # for ip in self.tello_ip_list:
        #     self.socket.sendto('land'.encode('utf-8'), (ip, 8889))
        # self.socket.close()

    def get_log(self):
        return self.log
