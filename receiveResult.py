import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 8080))
print("Listening for UDP messages on port 8080...")

while True:
    data, addr = sock.recvfrom(1024)
    mse_str, result, timestamp = data.decode().split(",")
    print(f"MSE: {mse_str}, Result: {result}, Time: {timestamp} ms")
