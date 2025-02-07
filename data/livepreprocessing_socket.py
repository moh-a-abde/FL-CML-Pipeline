import pandas as pd
import numpy as np
import warnings
import socket
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

# Ignore warnings
warnings.filterwarnings("ignore")

def read_kafka_topic(topic, bootstrap_servers):
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: loads(x.decode('utf-8'))
        )
        messages = []
        for message in consumer:
            messages.append(message.value)
            if len(messages) >= 10000:  # You can adjust the number of messages to consume
                break
        df = pd.DataFrame(messages)
        print("Messages consumed successfully!")
        return df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def clean(dfLocal):
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
    dfLocal.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    dfLocal = shuffle(dfLocal)
    return dfLocal

def summary(dfLocal):
    print(dfLocal.columns)
    print("number of columns", len(dfLocal.columns))
    print(dfLocal.head())
    print(dfLocal.shape)
    print(dfLocal.describe())
    print(dfLocal.describe(exclude=np.number))

port_label_mapping = {
    53: 'DNS',
    22: 'SSH',
    80: 'HTTP',
    443: 'HTTPS',
    21: 'FTP'
}

def get_label(port):
    return port_label_mapping.get(port, None)

def send_data_to_port(data, port=9000):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        sock.sendall(data.encode('utf-8'))
        sock.close()
        print("Data sent successfully!")
    except Exception as e:
        print(f"Failed to send data: {e}")

def process_data():
    topic = "zeek"
    bootstrap_servers = ["192.168.1.4:9092"]
    df = read_kafka_topic(topic, bootstrap_servers)
    
    if df is not None:
        print(df.columns)
        df = clean(df)
        summary(df)
        
        df['label'] = df['id.resp_p'].apply(lambda x: get_label(int(x)) if not pd.isna(x) else None)
        
        data = df.dropna(subset=['label'])
        
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
        
        data = data.drop(columns=drop_columns, errors='ignore')
        data.fillna(method='ffill', inplace=True)
        data = data.dropna()
        
        categorical_features = ['id.orig_h', 'id.resp_h', 'proto', 'history', 'uid', 'conn_state']
        numerical_features = ['id.orig_p', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'missed_bytes', 'local_resp',
                              'local_orig', 'resp_bytes', 'orig_bytes', 'duration', 'id.resp_p']
        
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
        
        X = data.drop(columns=['label', 'ts'])
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test).astype(str)
        
        print(classification_report(y_test, y_pred))

        # Send the final cleaned data to port 9000
        data_json = data.to_json(orient='records')
        send_data_to_port(data_json)

if __name__ == "__main__":
    try:
        while True:
            process_data()
    except KeyboardInterrupt:
        print("Process interrupted by user.")
