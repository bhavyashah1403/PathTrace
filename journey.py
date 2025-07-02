import os
import json
import pandas as pd
from datetime import datetime
import re


input_date = input("Enter date (DD-MM-YYYY): ").strip()
try:
    date_suffix = datetime.strptime(input_date, "%d-%m-%Y").strftime("%d-%m-%Y")
except ValueError:
    print("Invalid date format. Use DD-MM-YYYY")
    raise


logs_dir = os.path.join("logs", date_suffix)
os.makedirs(logs_dir, exist_ok=True)


df_transactions = pd.DataFrame()
df_ai_logs = pd.DataFrame()
df_comm_logs = pd.DataFrame()


timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} T \d{2}:\d{2}:\d{2}\.\d{3} Z)\]'

# --- Process Transaction Logs ---
log_files = [f for f in os.listdir('.') if f.startswith(f'Transaction-{date_suffix}') and f.endswith('.log')]
print(f"Found {len(log_files)} Transaction log file(s):", log_files)

parsed_records = []
skipped_lines = []
skipped_file_path = os.path.join(logs_dir, f"skipped_transaction_lines_{date_suffix}.log")

for file in log_files:
    with open(file, 'r', encoding='utf-8') as f:
        current_timestamp = None
        for line in f:
            line = line.strip()
            timestamp_match = re.match(timestamp_pattern, line)
            if timestamp_match:
                current_timestamp = timestamp_match.group(1)
                json_part = line[timestamp_match.end():].strip()
            else:
                json_part = line

            try:
                if not json_part:
                    continue
                entry = json.loads(json_part)

                inputdata_raw = entry.get('inputdata', '{}')
                inputdata = json.loads(inputdata_raw) if isinstance(inputdata_raw, str) else inputdata_raw
                responsedata_raw = entry.get('responsedata', '{}')
                responsedata = json.loads(responsedata_raw) if isinstance(responsedata_raw, str) else responsedata_raw
                scrip_info = inputdata.get('scrip_info', {})

                ucc_mobile_raw = entry.get('ucc-mobile', '')
                if '|' in ucc_mobile_raw:
                    ucc, mobile = ucc_mobile_raw.split('|', 1)
                elif ucc_mobile_raw.isdigit():
                    ucc, mobile = '', ucc_mobile_raw
                else:
                    ucc, mobile = ucc_mobile_raw, ''

                parsed_records.append({
                    'time': current_timestamp,
                    'service': entry.get('service'),
                    'statuscode': entry.get('statuscode'),
                    'accesstoken': entry.get('accesstoken'),
                    'authenticationtoken': entry.get('authenticationtoken'),
                    'taskname': inputdata.get('taskname'),
                    'order_type': inputdata.get('order_type'),
                    'product_type': inputdata.get('product_type'),
                    'transaction_type': inputdata.get('transaction_type'),
                    'price': inputdata.get('price'),
                    'quantity': inputdata.get('quantity'),
                    'second_auth': inputdata.get('second_auth'),
                    'symbol': scrip_info.get('symbol'),
                    'exchange': scrip_info.get('exchange'),
                    'scrip_token': scrip_info.get('scrip_token'),
                    'series': scrip_info.get('series'),
                    'expiry_date': scrip_info.get('expiry_date'),
                    'option_type': scrip_info.get('option_type'),
                    'response_status': responsedata.get('status'),
                    'response_message': responsedata.get('message'),
                    'order_id': responsedata.get('data', {}).get('orderId'),
                    'ucc': ucc,
                    'mobile': mobile,
                    'timetaken': entry.get('timetaken')
                })

            except Exception:
                skipped_lines.append(line)

if skipped_lines:
    with open(skipped_file_path, 'w', encoding='utf-8') as sf:
        for line in skipped_lines:
            sf.write(line + '\n')
    print(f"Skipped {len(skipped_lines)} Transaction line(s), saved to: {skipped_file_path}")
else:
    print("All Transaction lines parsed successfully.")

df_transactions = pd.DataFrame(parsed_records)

# --- Process StoxBot Logs ---
log_files = [f for f in os.listdir('.') if f.startswith(f'StoxBot-{date_suffix}') and f.endswith('.log')]
print(f"Found {len(log_files)} StoxBot log file(s):", log_files)

parsed_entries = []
skipped_lines = []
skipped_file = os.path.join(logs_dir, f'skipped_lines_{date_suffix}.log')

for file in log_files:
    with open(file, 'r', encoding='utf-8') as f:
        current_timestamp = None
        for line in f:
            line = line.strip()
            timestamp_match = re.match(timestamp_pattern, line)
            if timestamp_match:
                current_timestamp = timestamp_match.group(1)
                json_part = line[timestamp_match.end():].strip()
            else:
                json_part = line

            try:
                if not json_part:
                    continue
                record = json.loads(json_part)

                inputdata_raw = record.get('inputdata', {})
                if isinstance(inputdata_raw, str):
                    try:
                        inputdata_decoded = json.loads(inputdata_raw)
                    except json.JSONDecodeError:
                        inputdata_decoded = {}
                elif isinstance(inputdata_raw, dict):
                    inputdata_decoded = inputdata_raw
                else:
                    inputdata_decoded = {}

                ucc_mobile_raw = record.get('ucc-mobile', '')
                if '|' in ucc_mobile_raw:
                    ucc, mobile = ucc_mobile_raw.split('|', 1)
                elif ucc_mobile_raw.isdigit():
                    ucc, mobile = '', ucc_mobile_raw
                else:
                    ucc, mobile = ucc_mobile_raw, ''

                input_keyword_raw = inputdata_decoded.get('inputKeyword', '')
                response_text = ''
                if isinstance(input_keyword_raw, str) and input_keyword_raw.strip().startswith('['):
                    try:
                        keyword_list = json.loads(input_keyword_raw)
                        if isinstance(keyword_list, list) and keyword_list:
                            response_text = keyword_list[0].get('response', '')
                    except Exception:
                        response_text = ''
                elif isinstance(input_keyword_raw, str):
                    response_text = input_keyword_raw

                parsed_entries.append({
                    'time': current_timestamp,
                    'service': record.get('service'),
                    'statuscode': record.get('statuscode'),
                    'taskName': inputdata_decoded.get('taskName', ''),
                    'timetaken': record.get('timetaken'),
                    'ucc': ucc,
                    'mobile': mobile,
                    'inputKeyword_raw': json.dumps(input_keyword_raw) if isinstance(input_keyword_raw, (dict, list)) else str(input_keyword_raw),
                    'inputKeyword_response': response_text
                })

            except Exception:
                skipped_lines.append(line)

if skipped_lines:
    with open(skipped_file, 'w', encoding='utf-8') as sf:
        for skipped in skipped_lines:
            sf.write(skipped + '\n')
    print(f"Skipped {len(skipped_lines)} StoxBot line(s), saved to: {skipped_file}")
else:
    print("All StoxBot lines parsed successfully.")

df_ai_logs = pd.DataFrame(parsed_entries)

# --- Process StoxBotComm Logs ---
log_files = [f for f in os.listdir('.') if f.startswith(f'StoxBotComm-{date_suffix}') and f.endswith('.log')]
print(f"Found {len(log_files)} StoxBotComm log file(s):", log_files)

parsed_records = []
skipped_lines = []
skipped_file_path = os.path.join(logs_dir, f"skipped_comm_lines_{date_suffix}.log")

for file in log_files:
    with open(file, 'r', encoding='utf-8') as f:
        current_timestamp = None
        for line in f:
            line = line.strip()
            timestamp_match = re.match(timestamp_pattern, line)
            if timestamp_match:
                current_timestamp = timestamp_match.group(1)
                json_part = line[timestamp_match.end():].strip()
            else:
                json_part = line

            try:
                if not json_part:
                    continue
                record = json.loads(json_part)

                mo_raw = record.get('mo', '')
                mo_cleaned = mo_raw[2:] if mo_raw.startswith('91') else mo_raw

                st = record.get('st', '')
                tm = record.get('tm', '')
                if st == '' or tm == '':
                    parsed_records.append({
                        'time': current_timestamp,
                        'ct': record.get('ct', ''),
                        'id': record.get('id', ''),
                        'mo': mo_cleaned,
                        'st': st,
                        'tm': tm
                    })

            except Exception:
                skipped_lines.append(line)

if skipped_lines:
    with open(skipped_file_path, 'w', encoding='utf-8') as sf:
        for line in skipped_lines:
            sf.write(line + '\n')
    print(f"Skipped {len(skipped_lines)} StoxBotComm line(s), saved to: {skipped_file_path}")
else:
    print("All StoxBotComm lines parsed successfully.")

df_comm_logs = pd.DataFrame(parsed_records)

# --- Group Records by Number ---
df_transactions['second_auth'] = df_transactions['second_auth'].astype(str).str.strip()
df_transactions['mobile'] = df_transactions['mobile'].astype(str).str.strip()
df_ai_logs['mobile'] = df_ai_logs['mobile'].astype(str).str.strip()
df_comm_logs['mo'] = df_comm_logs['mo'].astype(str).str.strip()

all_numbers = pd.Series(pd.concat([
    df_transactions['second_auth'],
    df_transactions['mobile'],
    df_ai_logs['mobile'],
    df_comm_logs['mo']
])).dropna().astype(str).str.strip()

all_numbers = all_numbers[~all_numbers.isin(['', 'nan', 'NaN', 'None'])].unique()

output_lines = []
last_activity_records = []

for number in all_numbers:
    combined_records = []

    txn_logs = df_transactions[
        (df_transactions['second_auth'] == number) | (df_transactions['mobile'] == number)
    ]
    for _, txn_row in txn_logs.iterrows():
        combined_records.append({
            'time': txn_row.get('time', 'N/A'),
            'source': 'TRANSACTION',
            'key_field': txn_row.get('taskname', ''),
            'record': txn_row.to_dict()
        })

    stox_logs = df_ai_logs[df_ai_logs['mobile'] == number]
    for _, stox_row in stox_logs.iterrows():
        combined_records.append({
            'time': stox_row.get('time', 'N/A'),
            'source': 'STOXBOT',
            'key_field': stox_row.get('taskName', ''),
            'record': stox_row.to_dict()
        })

    comm_logs = df_comm_logs[df_comm_logs['mo'] == number]
    for _, comm_row in comm_logs.iterrows():
        combined_records.append({
            'time': comm_row.get('time', 'N/A'),
            'source': 'STOXBOTCOMM',
            'key_field': comm_row.get('st', ''),
            'record': comm_row.to_dict()
        })

    combined_records.sort(key=lambda x: x['time'] if x['time'] != 'N/A' else '9999-12-31 T 23:59:59.999 Z')

    # Find the last activity timestamp
    valid_records = [r for r in combined_records if r['time'] != 'N/A']
    if valid_records:
        last_record = valid_records[-1]
        last_activity_time = last_record['time']
        # Convert timestamp to YYYY-MM-DD HH:MM:SS format
        try:
            time_obj = datetime.strptime(last_activity_time, "%Y-%m-%d T %H:%M:%S.%f Z")
            formatted_time = time_obj.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_time = "N/A"
        last_activity_records.append({
            'mobile': number,
            'last_activity': formatted_time
        })

    output_lines.append(f"=== Records for Number: {number} ===\n")

    for record in combined_records:
        source = record['source']
        key_field = record['key_field']
        time_str = record['time']
        if time_str != 'N/A':
            time_str = time_str.replace(' T ', ' ').replace(' Z', '')
        if key_field:
            if source == 'STOXBOTCOMM':
                output_lines.append(f"**STATUS: {key_field} | TIME: {time_str}**")
            else:
                output_lines.append(f"**TASKNAME: {key_field} | TIME: {time_str}**")
        output_lines.append(f"[{source}] " + json.dumps(record['record'], default=str))

    output_lines.append("\n" + "-"*80 + "\n")


output_dir = os.path.join("linked_logs", date_suffix)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"grouped_records_by_number_{date_suffix}.txt")

with open(output_path, 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"Grouped records saved to: {output_path}")

# --- Generate CSV with Mobile Number and Last Activity Time ---
df_last_activity = pd.DataFrame(last_activity_records)
csv_output_path = os.path.join(output_dir, f"mobile_last_activity_{date_suffix}.csv")
df_last_activity.to_csv(csv_output_path, index=False)
print(f"Mobile number and last activity time CSV saved to: {csv_output_path}")
# Add to the end of the log processing script
df_transactions.to_csv(os.path.join(logs_dir, f"transactions_{date_suffix}.csv"), index=False)
df_ai_logs.to_csv(os.path.join(logs_dir, f"stoxbot_{date_suffix}.csv"), index=False)
df_comm_logs.to_csv(os.path.join(logs_dir, f"stoxbotcomm_{date_suffix}.csv"), index=False)