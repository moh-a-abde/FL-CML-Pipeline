import socket
import pandas as pd
import json
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("receiving_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def start_server(host='192.168.1.3', port=9000):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Set socket option to reuse address
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to the port
        server_socket.bind((host, port))
        
        # Listen for incoming connections
        server_socket.listen(5)
        logging.info(f"Server listening on {host}:{port}")
        
        while True:
            try:
                # Establish a connection
                logging.info("Waiting for connection...")
                client_socket, addr = server_socket.accept()
                logging.info(f"Got a connection from {addr}")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Receive the data in chunks
                data = b""
                chunks_received = 0
                
                while True:
                    try:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            break
                        chunks_received += 1
                        data += chunk
                        if chunks_received % 10 == 0:
                            logging.info(f"Received {chunks_received} chunks ({len(data)} bytes) so far")
                    except socket.timeout:
                        logging.warning("Socket timeout while receiving data")
                        break
                    except Exception as e:
                        logging.error(f"Error receiving data: {e}", exc_info=True)
                        break
                
                # Close the connection
                client_socket.close()
                logging.info(f"Connection closed. Received {len(data)} bytes in {chunks_received} chunks")
                
                if not data:
                    logging.warning("No data received from client")
                    continue
                
                # Decode the data to string
                try:
                    data_str = data.decode('utf-8')
                    logging.info(f"Data decoded successfully, length: {len(data_str)}")
                except UnicodeDecodeError as e:
                    logging.error(f"Failed to decode data: {e}", exc_info=True)
                    continue
                
                # Convert the JSON string to a Pandas DataFrame
                try:
                    logging.info("Attempting to parse JSON data")
                    df = pd.read_json(data_str, orient='records')
                    logging.info(f"JSON parsed successfully. DataFrame shape: {df.shape}")
                    
                    output_file_path = f'data/received_data_{timestamp}.csv'
                    df.to_csv(output_file_path, index=False)
                    logging.info(f"Data saved to {output_file_path}")
                    
                    # Also save a sample of the raw JSON for debugging
                    with open(f'data/raw_sample_{timestamp}.json', 'w') as f:
                        # Save just the first 1000 characters as a sample
                        f.write(data_str[:min(1000, len(data_str))])
                    logging.info(f"Raw sample saved to data/raw_sample_{timestamp}.json")
                    
                except ValueError as e:
                    logging.error(f"Failed to parse JSON: {e}", exc_info=True)
                    # Save the raw data for debugging
                    with open(f'data/invalid_json_{timestamp}.txt', 'w') as f:
                        f.write(data_str)
                    logging.info(f"Invalid JSON saved to data/invalid_json_{timestamp}.txt")
                except Exception as e:
                    logging.error(f"Failed to save data: {e}", exc_info=True)
            
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(f"Error handling connection: {e}", exc_info=True)
    
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}", exc_info=True)
    finally:
        server_socket.close()
        logging.info("Server socket closed")

if __name__ == "__main__":
    try:
        logging.info("Starting receiving_data.py")
        start_server()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
