import pandas as pd
import numpy as np
import warnings
import socket
import logging
import sys
from kafka import KafkaConsumer
from json import loads
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
import json
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
        logging.info(f"Attempting to connect to Kafka topic '{topic}' on {bootstrap_servers}")
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
                logging.info(f"Received {message_count} messages so far")
            messages.append(message.value)
            if len(messages) >= 10000:  # You can adjust the number of messages to consume
                break
        
        if not messages:
            logging.warning("No messages received from Kafka topic")
            return None
            
        logging.info(f"Total messages consumed: {len(messages)}")
        df = pd.DataFrame(messages)
        logging.info("Messages converted to DataFrame successfully!")
        return df
    except Exception as e:
        logging.error(f"An error occurred while reading from Kafka: {e}", exc_info=True)
        return None

def clean(dfLocal):
    try:
        if dfLocal.empty:
            logging.warning("DataFrame is empty, nothing to clean")
            return dfLocal
            
        logging.info(f"Initial DataFrame shape: {dfLocal.shape}")
        logging.info(f"Initial columns: {dfLocal.columns.tolist()}")
        
        drop_columns = [
            "peer", "metric_type", "prefix", "name", "labels",
            "label_values", "value", "mem", "pkts_proc", "events_proc", "events_queued",
            "bytes_recv", "pkts_dropped", "pkts_link", "pkts_lag", "active_tcp_conns",
            "active_udp_conns", "active_icmp_conns", "tcp_conns", "udp_conns", "icmp_conns",
            "timers", "active_timers", "files", "active_files", "dns_requests", "active_dns_requests",
            "reassem_tcp_size", "reassem_file_size", "reassem_frag_size", "reassem_unknown_size",
            "unit", "trans_id", "software_type", "version.major", "version.minor", "version.addl",
            "unparsed_version", "port_num", "port_proto", "ts_delta", "gaps", "ack", "percent_lost",
            "action", "size", "times.modified", "times.accessed", "times.created", "times.changed",
            "mode", "stratum", "poll", "precision", "root_delay", "root_disp", "ref_id", "ref_time",
            "org_time", "rec_time", "xmt_time", "num_exts", "notice", "source", "uids", "mac", "requested_addr",
            "msg_types", "host_name", "fingerprint", "certificate.version", "certificate.serial", "certificate.subject",
            "certificate.issuer", "certificate.not_valid_before", "certificate.not_valid_after", "certificate.key_alg",
            "certificate.sig_alg", "certificate.key_type", "certificate.key_length", "certificate.exponent",
            "san.dns", "basic_constraints.ca", "host_cert", "client_cert", "fuid", "depth", "analyzers", "mime_type",
            "acks", "is_orig", "seen_bytes", "total_bytes", "missing_bytes", "overflow_bytes", "timedout", "md5", "sha1",
            "extracted", "extracted_cutoff", "resp_fuids", "resp_mime_types", "cert_chain_fps", "client_cert_chain_fps",
            "subject", "issuer", "sni_matches_cert", "validation_status", "client_addr", "version.minor2", "host_p",
            "note", "msg", "sub", "src", "actions", "email_dest", "suppress_for", "direction", "level",
            "message", "location", "server_addr", "domain", "assigned_addr", "lease_time"
        ]
        
        # Check which columns actually exist before dropping
        existing_columns = [col for col in drop_columns if col in dfLocal.columns]
        logging.info(f"Dropping {len(existing_columns)} columns out of {len(drop_columns)} specified")
        
        dfLocal.drop(existing_columns, axis=1, inplace=True, errors='ignore')
        dfLocal = shuffle(dfLocal)
        
        logging.info(f"DataFrame shape after cleaning: {dfLocal.shape}")
        return dfLocal
    except Exception as e:
        logging.error(f"Error in clean function: {e}", exc_info=True)
        return dfLocal

def summary(dfLocal):
    try:
        logging.info("DataFrame columns: %s", dfLocal.columns.tolist())
        logging.info("Number of columns: %d", len(dfLocal.columns))
        logging.info("DataFrame head:\n%s", dfLocal.head().to_string())
        logging.info("DataFrame shape: %s", str(dfLocal.shape))
        logging.info("DataFrame numeric summary:\n%s", dfLocal.describe().to_string())
        logging.info("DataFrame non-numeric summary:\n%s", dfLocal.describe(exclude=np.number).to_string())
    except Exception as e:
        logging.error(f"Error in summary function: {e}", exc_info=True)

port_label_mapping = {
    53: 'DNS',
    22: 'SSH',
    80: 'HTTP',
    443: 'HTTPS',
    21: 'FTP'
}

def get_label(port):
    try:
        # Convert float to int if needed
        if isinstance(port, float):
            port = int(port)
        return port_label_mapping.get(port, None)
    except Exception as e:
        logging.error(f"Error in get_label function with port {port} (type: {type(port)}): {e}", exc_info=True)
        return None

def send_data_to_port(data, port=9000):
    try:
        logging.info(f"Attempting to send data to 192.168.1.3:{port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        sock.connect(('192.168.1.3', port))
        sock.sendall(data.encode('utf-8'))
        sock.close()
        logging.info("Data sent successfully!")
        return True
    except ConnectionRefusedError:
        logging.error(f"Connection refused to 192.168.1.3:{port}. Is the receiving server running?")
        return False
    except socket.timeout:
        logging.error(f"Connection to 192.168.1.3:{port} timed out")
        return False
    except Exception as e:
        logging.error(f"Failed to send data: {e}", exc_info=True)
        return False

def send_test_message():
    """Send a test message to verify socket communication."""
    try:
        logging.info("Sending test message to socket server")
        test_data = json.dumps([{"test": "message", "timestamp": datetime.now().isoformat()}])
        success = send_data_to_port(test_data)
        if success:
            logging.info("Test message sent successfully")
        else:
            logging.error("Failed to send test message")
        return success
    except Exception as e:
        logging.error(f"Error sending test message: {e}", exc_info=True)
        return False

def process_data():
    try:
        logging.info("Starting data processing")
        topic = "zeek"
        bootstrap_servers = ["192.168.1.3:9092"]
        
        # Generate a timestamp for this processing run
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file_path = f'received/network_traffic_{run_timestamp}.csv'
        logging.info(f"This run will save data to: {output_file_path}")
        
        # Create a list to collect all processed data
        all_processed_data = []
        
        df = read_kafka_topic(topic, bootstrap_servers)
        
        if df is None:
            logging.warning("No data received from Kafka, skipping processing")
            return False
            
        if df.empty:
            logging.warning("Received empty DataFrame from Kafka, skipping processing")
            return False
        
        logging.info(f"Initial DataFrame columns: {df.columns.tolist()}")
        df = clean(df)
        summary(df)
        
        # Check if 'id.resp_p' exists
        if 'id.resp_p' not in df.columns:
            logging.error("Column 'id.resp_p' not found in DataFrame")
            return False
            
        # Add logging for label creation
        logging.info(f"Unique values in 'id.resp_p' before labeling: {df['id.resp_p'].unique()}")
        logging.info(f"Data types in 'id.resp_p': {df['id.resp_p'].apply(type).unique()}")
        
        # Apply labels more safely
        df['label'] = df['id.resp_p'].apply(
            lambda x: get_label(x) if pd.notna(x) else None
        )
        
        logging.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        # Filter rows with valid labels
        data = df.dropna(subset=['label'])
        
        if data.empty:
            logging.warning("No data left after filtering for valid labels")
            # Instead of returning False, let's use all data and assign a default label
            logging.info("Using all data with a default 'UNKNOWN' label for ports not in mapping")
            df['label'] = df['id.resp_p'].apply(
                lambda x: get_label(x) if pd.notna(x) and get_label(x) is not None else 'UNKNOWN'
            )
            data = df
            logging.info(f"New label distribution: {data['label'].value_counts().to_dict()}")
        
        logging.info(f"DataFrame shape after label filtering: {data.shape}")
        
        drop_columns = [
            "version", "auth_attempts", "curve", "server_name", "resumed", "established", "ssl_history",
            "addl", "user_agent", "certificate.curve", "referrer", "host", "server", "status_msg",
            "cipher", "tags", "response_body_len", "status_code", "pkt_lag", "request_body_len",
            "uri", "service", "client", "mac_alg", "method", "trans_depth", "cipher_alg", "host_key",
            'rtt', 'query', 'qclass', 'qclass_name', 'qtype', 'qtype_name', 'rcode', 'rcode_name', 'AA', 'TC',
            'RD', 'RA', 'Z', 'answers', 'TTLs', 'rejected', 'compression_alg', 'kex_alg', 'host_key_alg',
            'auth_success', "orig_fuids", "orig_mime_types", "origin", "cause", "analyzer_kind", "analyzer_name",
            "failure_reason", "analyzer", "next_protocol", "id", "hashAlgorithm", "issuerNameHash",
            "issuerKeyHash", "serialNumber", "certStatus", "thisUpdate", "nextUpdate", "version.minor3",
            "last_alert", "proxied", "request_type", "till", "forwardable", "renewable", "cookie",
            "security_protocol", "cert_count", "dst", "p", "tunnel_type", "status", "request.host",
            "request_p", "bound.host", "bound_p", "client_scid", "failure_data", "san.ip", "resp_filenames"
        ]
        
        # Check which columns actually exist before dropping
        existing_columns = [col for col in drop_columns if col in data.columns]
        logging.info(f"Dropping {len(existing_columns)} columns out of {len(drop_columns)} specified in second drop")
        
        data = data.drop(columns=existing_columns, errors='ignore')
        data.fillna(method='ffill', inplace=True)
        data = data.dropna()
        
        if data.empty:
            logging.warning("No data left after dropping NA values")
            return False
            
        logging.info(f"DataFrame shape after NA dropping: {data.shape}")
        
        # Check if required features exist
        categorical_features = ['id.orig_h', 'id.resp_h', 'proto', 'history', 'uid', 'conn_state']
        numerical_features = ['id.orig_p', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'missed_bytes', 'local_resp',
                              'local_orig', 'resp_bytes', 'orig_bytes', 'duration', 'id.resp_p']
        
        missing_cat = [col for col in categorical_features if col not in data.columns]
        missing_num = [col for col in numerical_features if col not in data.columns]
        
        if missing_cat or missing_num:
            logging.error(f"Missing categorical features: {missing_cat}")
            logging.error(f"Missing numerical features: {missing_num}")
            # Use only available features
            categorical_features = [col for col in categorical_features if col in data.columns]
            numerical_features = [col for col in numerical_features if col in data.columns]
            
        if not categorical_features or not numerical_features:
            logging.error("No features available for model training")
            return False
            
        logging.info(f"Using categorical features: {categorical_features}")
        logging.info(f"Using numerical features: {numerical_features}")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])
        
        # Check if 'ts' exists
        drop_cols_for_X = ['label']
        if 'ts' in data.columns:
            drop_cols_for_X.append('ts')
            
        X = data.drop(columns=drop_cols_for_X)
        y = data['label']
        
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"Label distribution in training data: {y.value_counts().to_dict()}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Training model...")
        model.fit(X_train, y_train)
        
        logging.info("Predicting on test data...")
        y_pred = model.predict(X_test).astype(str)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        logging.info(f"Classification report:\n{json.dumps(report, indent=2)}")

        # Add processed data to our collection
        all_processed_data.append(data)
        
        # Combine all processed data
        if all_processed_data:
            final_data = pd.concat(all_processed_data, ignore_index=True)
            logging.info(f"Total processed data shape: {final_data.shape}")
            
            # Save all processed data to a single file for this run
            final_data.to_csv(output_file_path, index=False)
            logging.info(f"Saved all processed data to {output_file_path}")
            
            # Send the final cleaned data to port 9000
            data_json = final_data.to_json(orient='records')
            logging.info(f"JSON data size: {len(data_json)} bytes")
            success = send_data_to_port(data_json)
            
            if success:
                logging.info("Data processing and transmission completed successfully")
            else:
                logging.warning("Data processing completed but transmission failed")
                
            return success
        else:
            logging.warning("No data was processed")
            return False
    except Exception as e:
        logging.error(f"Error in process_data function: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    try:
        logging.info("Starting livepreprocessing_socket.py")
        
        # First, test socket communication
        if send_test_message():
            logging.info("Socket communication test successful, proceeding with data processing")
        else:
            logging.warning("Socket communication test failed, but proceeding anyway")
        
        # Run process_data only once instead of in a loop
        success = process_data()
        if success:
            logging.info("Data processing completed successfully. Exiting.")
        else:
            logging.error("Data processing failed. Exiting.")
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}", exc_info=True)
