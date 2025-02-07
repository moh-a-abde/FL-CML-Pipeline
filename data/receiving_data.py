import socket
import pandas as pd
import json
from datetime import datetime

def start_server(host='localhost', port=9000):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind to the port
    server_socket.bind((host, port))
    
    # Listen for incoming connections
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")
    
    while True:
        # Establish a connection
        client_socket, addr = server_socket.accept()
        print(f"Got a connection from {addr}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Receive the data in chunks
        data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            data += chunk
        
        # Close the connection
        client_socket.close()
        
        # Decode the data to string
        data_str = data.decode('utf-8')
        
        # Convert the JSON string to a Pandas DataFrame
        try:
            df = pd.read_json(data_str, orient='records')
            output_file_path = f'received_data_{timestamp}.csv'
            df.to_csv(output_file_path, index=False)
            print(f"Data saved to {output_file_path}")
        except Exception as e:
            print(f"Failed to save data: {e}")

if __name__ == "__main__":
    start_server()
