import pandas as pd
import numpy as np
import warnings
import socket
import logging
import sys
from kafka import KafkaConsumer
from json import loads
from sklearn.utils import shuffle
from datetime import datetime
import os

# Create received directory if it doesn't exist
os.makedirs('received', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("received/livepreprocessing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Ignore warnings
warnings.filterwarnings("ignore")

def read_kafka_topic(topic, bootstrap_servers):
    try:
        logging.info("Attempting to connect to Kafka topic '%s' on %s", topic, bootstrap_servers)
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            consumer_timeout_ms=30000  # 30 seconds timeout
        )
        messages = []
        logging.info("Connected to Kafka. Waiting for messages...")
        message_count = 0
        for message in consumer:
            message_count += 1
            if message_count % 100 == 0:
                logging.info("Received %d messages so far", message_count)
            messages.append(message.value)
            if len(messages) >= 10000:  # You can adjust the number of messages to consume
                break
            
        if not messages:
            logging.warning("No messages received from Kafka topic")
            return None
            
        logging.info("Total messages consumed: %d", len(messages))
        df = pd.DataFrame(messages)
        logging.info("Messages converted to DataFrame successfully!")
        return df
    except Exception as exc:
        logging.error("An error occurred while reading from Kafka: %s", exc, exc_info=True)
        return None

def clean(dfLocal):
    try:
        if dfLocal.empty:
            logging.warning("DataFrame is empty, nothing to clean")
            return dfLocal
            
        logging.info("Initial DataFrame shape: %s", dfLocal.shape)
        logging.info("Initial columns: %s", dfLocal.columns.tolist())
        
        # Drop non-essential columns
        drop_columns = [
            "peer", "metric_type", "prefix", "name", "labels",
            "label_values", "value", "mem", "events_proc", "events_queued",
            "timers", "active_timers", "files", "active_files", 
            "reassem_unknown_size", "unit", "trans_id", "software_type", 
            "version.major", "version.minor", "version.addl",
            "unparsed_version", "port_num", "port_proto", "gaps", "ack", 
            "action", "size", "times.modified", "times.accessed", "times.created", "times.changed",
            "mode", "stratum", "poll", "precision", "root_delay", "root_disp", "ref_id", "ref_time",
            "org_time", "rec_time", "xmt_time", "num_exts", "notice", "source", "uids", "mac", "requested_addr",
            "msg_types", "fingerprint", "certificate.version", "certificate.serial", "certificate.subject",
            "certificate.issuer", "certificate.not_valid_before", "certificate.not_valid_after", "certificate.key_alg",
            "certificate.sig_alg", "certificate.key_type", "certificate.key_length", "certificate.exponent",
            "san.dns", "basic_constraints.ca", "host_cert", "client_cert", "fuid", "depth", "analyzers", "mime_type",
            "resp_fuids", "resp_mime_types", "cert_chain_fps", "client_cert_chain_fps",
            "subject", "issuer", "sni_matches_cert", "client_addr", "version.minor2", "host_p",
            "note", "msg", "sub", "src", "actions", "email_dest", "suppress_for", "direction", "level",
            "message", "location", "assigned_addr", "lease_time", "host_name", "md5", "sha1", "extracted"
        ]
        
        
        # Check which columns actually exist before dropping
        existing_columns = [col for col in drop_columns if col in dfLocal.columns]
        logging.info("Dropping %d columns out of %d specified", len(existing_columns), len(drop_columns))
        
        dfLocal.drop(existing_columns, axis=1, inplace=True, errors='ignore')
        dfLocal = shuffle(dfLocal)
        
        logging.info("DataFrame shape after cleaning: %s", dfLocal.shape)
        return dfLocal
    except Exception as exc:
        logging.error("Error in clean function: %s", exc, exc_info=True)
        return dfLocal

def summary(dfLocal):
    try:
        logging.info("DataFrame columns: %s", dfLocal.columns.tolist())
        logging.info("Number of columns: %d", len(dfLocal.columns))
        logging.info("DataFrame head:\n%s", dfLocal.head().to_string())
        logging.info("DataFrame shape: %s", str(dfLocal.shape))
        logging.info("DataFrame numeric summary:\n%s", dfLocal.describe().to_string())
        logging.info("DataFrame non-numeric summary:\n%s", dfLocal.describe(exclude=np.number).to_string())
    except Exception as exc:
        logging.error("Error in summary function: %s", exc, exc_info=True)

def send_data_to_port(data, port=9000):
    try:
        data_size_mb = len(data) / (1024 * 1024)
        logging.info(f"Preparing to send {data_size_mb:.2f} MB of data to 192.168.1.3:{port}")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Increase socket buffer size
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)  # 8MB buffer
        current_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        logging.info(f"Socket send buffer size set to: {current_buffer_size / (1024*1024):.2f} MB")
        
        # Increase timeout for large data
        timeout_seconds = max(30, int(data_size_mb * 0.5))  # 0.5 seconds per MB, minimum 30 seconds
        sock.settimeout(timeout_seconds)
        logging.info(f"Socket timeout set to {timeout_seconds} seconds")
        
        sock.connect(('192.168.1.3', port))
        logging.info("Connected successfully, starting data transmission")
        
        # Send in chunks
        chunk_size = 1048576  # 1MB chunks
        total_sent = 0
        data_encoded = data.encode('utf-8')
        total_size = len(data_encoded)
        
        while total_sent < total_size:
            chunk = data_encoded[total_sent:total_sent + chunk_size]
            sent = sock.send(chunk)
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            total_sent += sent
            if total_sent % (50 * 1048576) == 0:  # Log every 50MB
                logging.info(f"Progress: {(total_sent/total_size)*100:.2f}% ({total_sent/(1024*1024):.2f}MB / {total_size/(1024*1024):.2f}MB)")
        
        logging.info(f"Data transmission completed. Total sent: {total_sent/(1024*1024):.2f}MB")
        sock.close()
        return True
    except ConnectionRefusedError:
        logging.error("Connection refused to 192.168.1.3:%d. Is the receiving server running?", port)
        return False
    except socket.timeout:
        logging.error(f"Connection/transmission to 192.168.1.3:{port} timed out after {timeout_seconds} seconds")
        logging.error(f"Only sent {total_sent/(1024*1024):.2f}MB out of {total_size/(1024*1024):.2f}MB")
        return False
    except Exception as exc:
        logging.error(f"Failed to send data: {exc}", exc_info=True)
        return False

import pandas as pd
import logging
from datetime import datetime

def process_data():
    try:
        logging.info("Starting data processing")
        topic = "zeek"
        bootstrap_servers = ["192.168.1.3:9092"]
        
        # Generate a timestamp for this processing run
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file_path = f'received/network_traffic_{run_timestamp}.csv'
        logging.info("This run will save data to: %s", output_file_path)
        
        # Create a list to collect all processed data
        all_processed_data = []
        
        df = read_kafka_topic(topic, bootstrap_servers)
        
        if df is None:
            logging.warning("No data received from Kafka, skipping processing")
            return False
            
        if df.empty:
            logging.warning("Received empty DataFrame from Kafka, skipping processing")
            return False
        
        logging.info("Initial DataFrame columns: %s", df.columns.tolist())
        df = clean(df)
        summary(df)
        
        # Make a copy of df to avoid modifying the original
        data = df.copy()
        
        # Drop additional non-essential columns
        drop_columns = [
            "version", "curve", "server_name", "resumed", "established", "ssl_history",
            "addl", "user_agent", "certificate.curve", "referrer", "host", "server", 
            "cipher", "tags", "uri", "service", "client", "host_key",
            "query", "qclass", "qclass_name", "qtype", "qtype_name", "rcode", "rcode_name", "AA", "TC",
            "RD", "RA", "Z", "answers", "TTLs", "rejected", "compression_alg", "host_key_alg",
            "orig_fuids", "orig_mime_types", "origin", "cause", "analyzer_kind", "analyzer_name",
            "failure_reason", "analyzer", "next_protocol", "id", "hashAlgorithm", "issuerNameHash",
            "issuerKeyHash", "serialNumber", "certStatus", "thisUpdate", "nextUpdate", "version.minor3",
            "last_alert", "proxied", "request_type", "till", "forwardable", "renewable", "cookie",
            "cert_count", "dst", "p", "tunnel_type", "status", "request.host",
            "request_p", "bound.host", "bound_p", "client_scid", "failure_data", "san.ip", "resp_filenames"
        ]

        # Check which columns actually exist before dropping
        existing_columns = [col for col in drop_columns if col in data.columns]
        logging.info("Dropping %d columns out of %d specified in second drop", len(existing_columns), len(drop_columns))
        
        data = data.drop(columns=existing_columns, errors='ignore')
        data.fillna(method='ffill', inplace=True)
        data = data.dropna()
        
        if data.empty:
            logging.warning("No data left after dropping NA values")
            return False
            
        logging.info("DataFrame shape after NA dropping: %s", data.shape)
        
        # Add processed data to our collection
        all_processed_data.append(data)
        
        # Combine all processed data
        if not all_processed_data:
            logging.warning("No data was processed")
            return False
            
        final_data = pd.concat(all_processed_data, ignore_index=True)
        logging.info("Total processed data shape: %s", final_data.shape)
        
        # **Add Labels for Classification**
        final_data["label"] = 0  # Default label as 0
        final_data.loc[final_data.index[-10:], "label"] = 1  # Set last 10 rows to 1
        
        # Save all processed data to a single file for this run
        final_data.to_csv(output_file_path, index=False)
        logging.info("Saved all processed data to %s", output_file_path)
        
        # Send the final cleaned data to port 9000
        data_json = final_data.to_json(orient='records')
        logging.info("JSON data size: %d bytes", len(data_json))
        send_result = send_data_to_port(data_json)
        
        if send_result:
            logging.info("Data processing and transmission completed successfully")
        else:
            logging.warning("Data processing completed but transmission failed")
            
        return send_result
        
    except Exception as exc:
        logging.error("Error in process_data function: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    try:
        logging.info("Starting livepreprocessing_socket.py")
     
        # Run process_data only once instead of in a loop
        success = process_data()
        if success:
            logging.info("Data processing completed successfully. Exiting.")
        else:
            logging.error("Data processing failed. Exiting.")
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    except Exception as exc:
        logging.error("Unhandled exception in main: %s", exc, exc_info=True)
