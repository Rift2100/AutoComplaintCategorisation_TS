
import json
import re
from groq import Groq

class EnhancedComplaintCategorizer:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        
        #the complaint categories
        self.categories = [
            "Delay in issuing caste certificate",
            "Land mutation/name correction pending", 
            "Ration shop short supply/overpricing",
            "Ration card member addition not updated",
            "MGNREGS wage not credited",
            "Job card issue/muster discrepancy", 
            "Handpump/borwell not working",
            "Pipeline leak/irregular water supply",
            "Frequent power outages/transformer failure",
            "Billing dispute/meter error",
            "School MDM quality complaint",
            "PHC medicine stock-out",
            "Seed/fertilizer quality issue",
            "Crop damage assessment request",
            "Streetlights not working",
            "Village road/drain repair needed"
        ]
        
        # keyword mappings for formal letters
        self.keyword_mappings = {
            "Land mutation/name correction pending": [
                "land mutation", "mutation", "survey no", "agricultural land", 
                "tahsildar", "land records", "patta", "grievance redressal portal"
            ],
            "Ration card member addition not updated": [
                "ration card", "member addition", "deletion", "update", "e-kyc"
            ],
            "Ration shop short supply/overpricing": [
                "ration shop", "fps", "fair price shop", "dealer", "rice", "wheat", 
                "supply officer", "overpricing", "short supply", "distribution"
            ],
            "MGNREGS wage not credited": [
                "mgnregs", "wage", "nrega", "job card", "fto", "rural development",
                "programme officer", "wage credited"
            ],
            "Frequent power outages/transformer failure": [
                "power outages", "electricity", "transformer", "apspdcl", "power supply",
                "electrical", "current", "feeder", "outage"
            ],
            "Pipeline leak/irregular water supply": [
                "water supply", "pipeline", "handpump", "water", "leak", "boring", 
                "rwss", "irregular supply"
            ],
            "Crop damage assessment request": [
                "farmer", "crop failure", "go-43", "compensation", "district collector",
                "agricultural", "crop damage", "drought", "flood"
            ],
            "School MDM quality complaint": [
                "school", "midday meal", "mdm", "meal quality", "education officer",
                "children", "food quality"
            ],
            "PHC medicine stock-out": [
                "phc", "medicine", "health", "medical officer", "stock", "hospital",
                "healthcare", "medicines"
            ],
            "Delay in issuing caste certificate": [
                "caste certificate", "certificate", "caste", "sc", "st", "obc", "bc"
            ],
            "Streetlights not working": [
                "streetlight", "street light", "lighting", "lamp", "municipal"
            ],
            "Village road/drain repair needed": [
                "road", "drain", "repair", "pothole", "village road", "maintenance"
            ]
        }

    def enhanced_keyword_match(self, complaint_text):
        text_lower = complaint_text.lower()
        scores = {}
        
        for category, keywords in self.keyword_mappings.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 2
                elif any(word in text_lower for word in keyword.split()):
                    score += 1
            
            if score > 0:
                scores[category] = score
        
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            confidence = min(best_category[1] / 5.0, 1.0)  # Normalize to 0-1
            return best_category[0], confidence
        
        return None, 0

    def simple_llm_categorize(self, complaint_text):
        """Simple LLM categorization with focused prompt using Llama 3.1"""
        try:
            if "Subject:" in complaint_text and "Respected Sir/Madam" in complaint_text:
                lines = complaint_text.split('\n')
                subject = ""
                content_lines = []
                
                for line in lines:
                    if line.strip().startswith("Subject:"):
                        subject = line.replace("Subject:", "").strip()
                        break
                
                content_started = False
                for line in lines:
                    if "Respected Sir/Madam" in line:
                        content_started = True
                        continue
                    if content_started and line.strip():
                        content_lines.append(line.strip())
                        if len(content_lines) >= 2:
                            break
                
                content = " ".join(content_lines)
                text_to_analyze = f"{subject} {content}"
            else:
                # Simple complaint format
                text_to_analyze = complaint_text.strip()
            
            prompt = f"""What category number fits this complaint?

1=caste certificate, 2=land mutation, 3=ration shop, 4=ration card, 5=MGNREGS wage, 6=job card, 7=handpump, 8=water pipeline, 9=power outage, 10=billing, 11=school meal, 12=medicine, 13=fertilizer, 14=crop damage, 15=streetlight, 16=road repair

Complaint: "{text_to_analyze[:200]}"

Answer (just the number):"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a complaint categorizer. Return only a number 1-12."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=5
            )
            
            result = response.choices[0].message.content.strip()
            
            category_map = {
                "1": "Delay in issuing caste certificate",
                "2": "Land mutation/name correction pending",
                "3": "Ration shop short supply/overpricing",
                "4": "Ration card member addition not updated",
                "5": "MGNREGS wage not credited",
                "6": "Job card issue/muster discrepancy",
                "7": "Handpump/borwell not working",
                "8": "Pipeline leak/irregular water supply",
                "9": "Frequent power outages/transformer failure",
                "10": "Billing dispute/meter error",
                "11": "School MDM quality complaint",
                "12": "PHC medicine stock-out",
                "13": "Seed/fertilizer quality issue",
                "14": "Crop damage assessment request",
                "15": "Streetlights not working",
                "16": "Village road/drain repair needed"
            }
            
            return category_map.get(result, "General Complaint")
            
        except Exception as e:
            print(f"LLM error: {e}")
            return "General Complaint"

    def categorize_complaint(self, complaint_text):
        category, confidence = self.enhanced_keyword_match(complaint_text)
        
        if category and confidence > 0.5:
            print(f"Keyword match: {category} (confidence: {confidence:.2f})")
            return category

        print("Using LLM for categorization...")
        llm_category = self.simple_llm_categorize(complaint_text)
        if category and confidence > 0.2:
            print(f"Keyword: {category}, LLM: {llm_category}")
            if llm_category == "General Complaint":
                return category
        
        return llm_category
