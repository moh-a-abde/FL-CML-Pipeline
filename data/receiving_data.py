import socket
import pandas as pd
import json
import logging
import sys
import os
from datetime import datetime
import io

# Create received directory if it doesn't exist
os.makedirs('received', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("received/receiving_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def start_server(host='192.168.1.3', port=9000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Set socket option to reuse address and increase buffer size
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)  # 8MB buffer
        current_buffer_size = server_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        logging.info(f"Socket receive buffer size set to: {current_buffer_size / (1024*1024):.2f} MB")
        
        # Bind to the port
        server_socket.bind((host, port))
        server_socket.listen(5)
        logging.info(f"Server listening on {host}:{port}")
        
        while True:
            try:
                logging.info("Waiting for connection...")
                client_socket, addr = server_socket.accept()
                logging.info(f"Got a connection from {addr}")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Set a longer timeout for receiving data
                client_socket.settimeout(120)  # 2 minutes timeout
                
                # Receive the data in chunks
                data = bytearray()
                chunks_received = 0
                start_time = datetime.now()
                last_log_time = start_time
                
                while True:
                    try:
                        chunk = client_socket.recv(1048576)  # 1MB chunks
                        if not chunk:
                            break
                        chunks_received += 1
                        data.extend(chunk)
                        
                        # Log progress every 5 seconds or 50MB
                        current_time = datetime.now()
                        if (current_time - last_log_time).seconds >= 5 or len(data) % (50 * 1048576) < 1048576:
                            mb_received = len(data) / (1024 * 1024)
                            mb_per_second = mb_received / (current_time - start_time).total_seconds()
                            logging.info(f"Received {mb_received:.2f}MB in {chunks_received} chunks. Transfer rate: {mb_per_second:.2f}MB/s")
                            last_log_time = current_time
                            
                    except socket.timeout:
                        logging.warning("Socket timeout while receiving data")
                        break
                    except Exception as e:
                        logging.error(f"Error receiving data: {e}", exc_info=True)
                        break
                
                # Close the connection
                client_socket.close()
                total_mb = len(data) / (1024 * 1024)
                duration = (datetime.now() - start_time).total_seconds()
                avg_speed = total_mb / duration if duration > 0 else 0
                logging.info(f"Connection closed. Received {total_mb:.2f}MB in {chunks_received} chunks. Average speed: {avg_speed:.2f}MB/s")
                
                if not data:
                    logging.warning("No data received from client")
                    continue
                
                # Decode the data to string
                try:
                    data_str = data.decode('utf-8')
                    logging.info(f"Data decoded successfully, length: {len(data_str)}")
                    
                    # Validate JSON structure
                    if not (data_str.startswith('[') and data_str.endswith(']')):
                        logging.error("Invalid JSON structure: Data doesn't start with '[' and end with ']'")
                        logging.info(f"Data starts with: {data_str[:100]}")
                        logging.info(f"Data ends with: {data_str[-100:] if len(data_str) > 100 else data_str}")
                    
                except UnicodeDecodeError as e:
                    logging.error(f"Failed to decode data: {e}", exc_info=True)
                    continue
                
                # Convert the JSON string to a Pandas DataFrame
                try:
                    logging.info("Attempting to parse JSON data")
                    # Use StringIO to avoid FutureWarning
                    df = pd.read_json(io.StringIO(data_str), orient='records')
                    logging.info(f"JSON parsed successfully. DataFrame shape: {df.shape}")
                    
                    # Save to a file with timestamp
                    output_file_path = f'received/received_data_{timestamp}.csv'
                    df.to_csv(output_file_path, index=False)
                    logging.info(f"Data saved to {output_file_path}")
                    
                except ValueError as e:
                    logging.error(f"Failed to parse JSON: {e}", exc_info=True)
                    # Save both the start and end of the raw data for debugging
                    with open(f'received/invalid_json_{timestamp}.txt', 'w', encoding='utf-8') as f:
                        f.write("=== First 1000 characters ===\n")
                        f.write(data_str[:1000])
                        f.write("\n\n=== Last 1000 characters ===\n")
                        f.write(data_str[-1000:])
                    logging.info(f"Invalid JSON sample saved to received/invalid_json_{timestamp}.txt")
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
        logging.info("Starting receiving server...")
        start_server()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
