
import socket
import time

TELLO_IP = "192.168.10.1"
TELLO_PORT = 8889

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", 8889))
sock.settimeout(2)

def send(command, timeout=15):
    print(f"Sending: {command}")
    sock.sendto(
        command.encode("ascii"),
        (TELLO_IP, TELLO_PORT)
    )

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue

        if addr[0] != TELLO_IP or addr[1] != TELLO_PORT:
            continue

        try:
            response = data.decode("ascii").strip()
        except UnicodeDecodeError:
            continue

        print(f"Response: {response}")
        return response

    print("No response")
    return None


takeoff_attempted = False

try:
    if send("command") != "ok":
        raise RuntimeError("SDK initialization failed")

    battery = send("battery?")
    if battery is None or not battery.isdigit():
        raise RuntimeError("Battery check failed")

    print(f"Battery: {battery}%")

    if int(battery) < 30:
        raise RuntimeError("Battery too low")

    print("Taking off...")
    takeoff_attempted = True
    result = send("takeoff", timeout=20)

    if result != "ok":
        raise RuntimeError("Takeoff not confirmed")

    print("Hovering for 5 seconds...")
    time.sleep(5)

finally:
    if takeoff_attempted:
        print("Landing...")
        send("land", timeout=15)

    sock.close()
