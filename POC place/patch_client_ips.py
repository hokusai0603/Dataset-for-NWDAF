import os
import glob
import json
import pandas as pd

directories = ['cat123_training_data', 'cat1_cat2_training_data']

ip_mapping = {
    '10.0.0.1': '10.10.0.1',
    '10.0.0.2': '10.10.0.2',
    '10.0.0.3': '10.10.0.3'
}

def patch_datasets():
    for folder in directories:
        if not os.path.exists(folder):
            print(f"Skipping {folder}, does not exist.")
            continue
        
        # Patch labels parquet
        for file in glob.glob(os.path.join(folder, 'training_labels_*.parquet')):
            df = pd.read_parquet(file)
            if 'ue_ip' in df.columns:
                df['ue_ip'] = df['ue_ip'].replace(ip_mapping)
            df.to_parquet(file, index=False)
            print(f"Patched labels: {file}")
            
        # Patch packets parquet
        for file in glob.glob(os.path.join(folder, 'training_packets_*.parquet')):
            df = pd.read_parquet(file)
            if 'ue_ip' in df.columns:
                df['ue_ip'] = df['ue_ip'].replace(ip_mapping)
            if 'src_ip' in df.columns:
                df['src_ip'] = df['src_ip'].replace(ip_mapping)
            if 'dst_ip' in df.columns:
                df['dst_ip'] = df['dst_ip'].replace(ip_mapping)
            df.to_parquet(file, index=False)
            print(f"Patched packets: {file}")

        # Patch notifications json
        for file in glob.glob(os.path.join(folder, 'training_notifications_*.json')):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modified = False
            for notif in data:
                if 'notificationItems' in notif:
                    for item in notif['notificationItems']:
                        if 'ueIpv4Addr' in item and item['ueIpv4Addr'] in ip_mapping:
                            item['ueIpv4Addr'] = ip_mapping[item['ueIpv4Addr']]
                            modified = True
            
            if modified:
                with open(file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                print(f"Patched notifications: {file}")

if __name__ == '__main__':
    patch_datasets()
    print("Patching completed.")
