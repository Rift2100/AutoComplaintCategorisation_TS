
import pandas as pd
import os
from datetime import datetime
import json

class ComplaintTracker:
    def __init__(self, storage_file="submitted_complaints.csv"):
        self.storage_file = storage_file
        self.ensure_storage_file()
    
    def ensure_storage_file(self):
        if not os.path.exists(self.storage_file):
            columns = [
                'complaint_number', 'name', 'phone', 'aadhaar', 'address', 
                'complaint_text', 'complaint_type', 'assignee', 'sla_date', 
                'sla_days', 'support_level_1', 'support_level_2', 'status',
                'submission_time', 'mandal', 'village'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.storage_file, index=False)
    
    def add_complaint(self, complaint_data):
        try:
            df = pd.read_csv(self.storage_file)
            
            address = complaint_data.get('address', '')
            parts = address.split(',')
            mandal = parts[-1].strip().replace('Mandal', '').strip() if len(parts) > 1 else 'Unknown'
            village = parts[-2].strip() if len(parts) > 2 else 'Unknown'
            
            new_complaint = {
                'complaint_number': complaint_data['complaint_number'],
                'name': complaint_data['name'],
                'phone': complaint_data['phone'],
                'aadhaar': complaint_data['aadhaar'],
                'address': complaint_data['address'],
                'complaint_text': complaint_data.get('complaint_text', '')[:500],  # Truncate for CSV
                'complaint_type': complaint_data['complaint_type'],
                'assignee': complaint_data['assignee'],
                'sla_date': complaint_data['sla_date'],
                'sla_days': complaint_data['sla_days'],
                'support_level_1': complaint_data['support_levels'][0] if complaint_data['support_levels'][0] else '',
                'support_level_2': complaint_data['support_levels'][1] if complaint_data['support_levels'][1] else '',
                'status': 'Registered',
                'submission_time': complaint_data['time'],
                'mandal': mandal,
                'village': village
            }
            
            new_row = pd.DataFrame([new_complaint])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.storage_file, index=False)
            
            return True
            
        except Exception as e:
            print(f"Error adding complaint: {e}")
            return False
    
    def search_complaints(self, search_method, search_value):
        try:
            if not os.path.exists(self.storage_file):
                return []
                
            df = pd.read_csv(self.storage_file)
            
            if len(df) == 0:
                return []
            
            results = []
            
            if search_method == "Complaint Number":
                matches = df[df['complaint_number'].astype(str).str.contains(search_value, case=False, na=False)]
                
            elif search_method == "Phone Number":
                phone_clean = search_value.strip()
                matches = df[df['phone'].astype(str).str.contains(phone_clean, na=False)]
                
            elif search_method == "Aadhaar Number":
                aadhaar_last4 = search_value.strip()[-4:] if len(search_value.strip()) >= 4 else search_value.strip()
                matches = df[df['aadhaar'].astype(str).str.endswith(aadhaar_last4, na=False)]
                
            else:
                return []

            if len(matches) > 0:
                results = matches.to_dict('records')
            
            return results
            
        except Exception as e:
            print(f"Error searching complaints: {e}")
            return []
    
    def get_complaint_stats(self):
        try:
            if not os.path.exists(self.storage_file):
                return {"total": 0, "by_status": {}}
            
            df = pd.read_csv(self.storage_file)
            
            stats = {
                "total": len(df),
                "by_status": df['status'].value_counts().to_dict() if len(df) > 0 else {},
                "by_type": df['complaint_type'].value_counts().head(5).to_dict() if len(df) > 0 else {}
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total": 0, "by_status": {}}
