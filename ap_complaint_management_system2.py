# AP Mandal Complaint Management System (Smart Voice + Document OCR Auto-Fill)
# Latest change: Added WAV upload option and text processing for voice complaints
# Comments: Three auto-fill methods - voice recording OR WAV upload OR text input + document upload
# Dependencies: gradio, pandas, groq, numpy, speech_recognition, google-generativeai, pillow

import gradio as gr
import pandas as pd
import numpy as np
import datetime
import re
import os
from llm_complaint_categorizer_v2 import EnhancedComplaintCategorizer
from complaint_tracker import ComplaintTracker
import speech_recognition as sr

from PIL import Image
import io
import base64
from dotenv import load_dotenv
import traceback

from pathlib import Path
import json
from groq import Groq
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from sarvamai import SarvamAI

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
llm_categorizer = EnhancedComplaintCategorizer(GROQ_API_KEY)
tracker = ComplaintTracker()

# --- Document OCR Processing Functions ---
def process_single_image_ocr(model, image: Image.Image, target_language: str = "English"):
    """Process a single image for OCR and translation."""
    try:
        instruction = (
            "First, perform Optical Character Recognition (OCR) on this image to extract ALL text. "
            "The text may be in Telugu, Hindi, Tamil, Kannada, and English Indian languages. "
            "Second, translate all extracted text into English if it's not already in English. "
            "Return ONLY the extracted and translated text, maintaining the original structure and formatting."
        )

        response = model.generate_content(
            [instruction, image],
            generation_config=GenerationConfig(temperature=0.2)
        )

        result = response.text.strip()
        return result

    except Exception as e:
        raise Exception(f"OCR processing error: {str(e)}")

def process_smart_document_input(document_file):
    if document_file is None:
        return "No document uploaded", "", "", "", "", "", ""
    
    try:
        if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
            return "Error: Gemini API key not found. Please add GEMINI_API_KEY to .env file", "", "", "", "", "", ""
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        print(f"Processing document: {document_file}")
        try:
            img = Image.open(document_file)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
        except Exception as e:
            return f"Error loading image: {str(e)}", "", "", "", "", "", ""

        extracted_text = process_single_image_ocr(model, img)
        print(f"OCR extracted text: {extracted_text[:200]}...")
        
        if not extracted_text.strip():
            return "No text could be extracted from the document. Please upload a clearer image.", "", "", "", "", "", ""

        extracted_info = extract_info_from_document(extracted_text)

        # print(f"Extracted Name: {extracted_info.get('name', 'Not found')}")
        # print(f"Extracted Phone: {extracted_info.get('phone', 'Not found')}")
        # print(f"Extracted Address: {extracted_info.get('address', 'Not found')}")
        # print(f"Complaint Text: {extracted_info.get('complaint_text', '')[:100]}...")

        complaint_text = extracted_info.get('complaint_text', extracted_text)
        # print(f"Classifying complaint: {complaint_text[:100]}...")
        complaint_type = llm_categorizer.categorize_complaint(complaint_text)
        print(f"Auto-classified as: {complaint_type}")
        
        # Create classification status message
        classification_status = f"**Auto-Classified Complaint Type:** {complaint_type}"
        
        return (
            complaint_text,
            extracted_info.get('name', ''),
            extracted_info.get('phone', ''),
            extracted_info.get('aadhaar', ''),
            extracted_info.get('address', ''),
            classification_status,
            complaint_type
        )
        
    except Exception as e:
        print(f"Document processing error: {e}")
        return f"Error processing document: {str(e)[:100]}...", "", "", "", "", "", ""

def extract_info_from_document(text):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        extraction_prompt = f"""
Extract information from this text (may be from voice or document). Extract ALL available information.

Text: "{text}"

Return ONLY in this exact JSON format:
{{
    "name": "extracted full name or empty string",
    "phone": "9-10 digit phone number or empty string", 
    "aadhaar": "12-digit aadhaar or empty string",
    "address": "full address or empty string",
    "complaint_text": "the actual complaint description"
}}

CRITICAL NAME EXTRACTION RULES:
1. IGNORE names that appear after "To", "Director", "Officer", "Chief", at the START of letter
2. EXTRACT names that appear at the END of letter after:
   - "Yours faithfully," "Respectfully," "Submitted by"
   - Signatures section at bottom
   - Names in parentheses like (John Doe) at end
3. For formal letters: Look for sender's name at BOTTOM, not recipient at TOP
4. Multiple names at end: Extract the first clear name in parentheses or after closing

NAME PATTERNS:
- English: "My name is John", "I am John", "This is John"
- Hindi: "mera naam hai Rohit", "main Rohit hoon"
- Telugu: "naa peru", "nenu"
- Letter closing: "Yours faithfully, [Name]", "(Name)", names after signatures

PHONE NUMBER PATTERNS:
- 9 or 10 digits starting with 6-9: 987654321, 9876543210, 98765 43210
- Mobile: 9876543210, Contact: 987654321
- Hindi: "mera number hai nau aath saat"
- Convert Hindi numbers: ek=1, do=2, teen=3, char=4, panch=5, cheh=6, saat=7, aath=8, nau=9, das=0
- Accept both 9 and 10 digit formats

ADDRESS PATTERNS:
- D.No., H.No., Village, Mandal, District
- "address", "pata", "ghar"
- Location at letter end or in body

AADHAAR PATTERNS:
- 12-digit numbers, "aadhaar", "aadhar card"

COMPLAINT PATTERNS:
- Extract from "Subject:" line and main body
- Main issue: power cut, road, water, land, ration, wages, etc.
- Ignore "To" and "Date" sections
- Focus on complaint description in body

IMPORTANT:
- For formal letters: sender name is at BOTTOM, recipient at TOP - extract BOTTOM name
- Skip titles like "Director", "Officer", "Chief Minister"
- Extract actual person names, not designations
"""
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        # print(f"LLM extraction result: {result_text}")
        
        try:
            extracted = json.loads(result_text)
            
            validated_info = {
                'name': clean_name(extracted.get('name', '')),
                'phone': clean_phone(extracted.get('phone', '')),
                'aadhaar': clean_aadhaar(extracted.get('aadhaar', '')),
                'address': clean_address(extracted.get('address', '')),
                'complaint_text': extracted.get('complaint_text', text)
            }
            
            return validated_info
            
        except json.JSONDecodeError:
            print("Failed to parse LLM JSON response")
            return fallback_extraction(text)
            
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return fallback_extraction(text)

def process_text_input(text_input):
    if not text_input or not text_input.strip():
        return "Please enter complaint text", "", "", "", "", ""
    
    try:
        print(f"Processing text input: {text_input[:100]}...")

        extracted_info = extract_info_from_speech(text_input)
        
        # print(f"Extracted Name: {extracted_info.get('name', 'Not found')}")
        # print(f"Extracted Phone: {extracted_info.get('phone', 'Not found')}")
        # print(f"Extracted Address: {extracted_info.get('address', 'Not found')}")
        # print(f"Complaint Text: {extracted_info.get('complaint_text', '')[:100]}...")
        
        complaint_text = extracted_info.get('complaint_text', text_input)
        # print(f"Classifying complaint: {complaint_text[:100]}...")
        complaint_type = llm_categorizer.categorize_complaint(complaint_text)
        print(f"Auto-classified as: {complaint_type}")
        
        # Create classification status message
        classification_status = f"**Auto-Classified Complaint Type:** {complaint_type}"
        
        return (
            complaint_text,
            extracted_info.get('name', ''),
            extracted_info.get('phone', ''),
            extracted_info.get('aadhaar', ''),
            extracted_info.get('address', ''),
            classification_status
        )
        
    except Exception as e:
        print(f"Text processing error: {e}")
        return f"Error: {str(e)[:50]}...", "", "", "", "", ""

def _original_process_smart_voice_input(audio_file):
    """Convert voice to text, extract structured information and classify complaint"""
    if audio_file is None:
        return "No audio recorded", "", "", "", "", ""
    
    try:
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        
        # print(f"Processing smart voice input: {audio_file}")
        
        with sr.AudioFile(audio_file) as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = r.record(source)

        languages = ['te','hi', 'en']
        full_text = ""
        
        for lang in languages:
            try:
                full_text = r.recognize_groq(audio_data, language=lang)
                if full_text.strip():
                    print(f"Voice recognized: {full_text}")
                    break
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                continue
        
        if not full_text.strip():
            return "Speech not recognized. Please speak clearly or type manually.", "", "", "", "", ""

        extracted_info = extract_info_from_speech(full_text)
        
        complaint_text = extracted_info.get('complaint_text', full_text)
        # print(f"Classifying complaint: {complaint_text[:100]}...")
        complaint_type = llm_categorizer.categorize_complaint(complaint_text)
        print(f"Auto-classified as: {complaint_type}")
        
        # Create classification status message
        classification_status = f"**Auto-Classified Complaint Type:** {complaint_type}"
        
        return (
            complaint_text,
            extracted_info.get('name', ''),
            extracted_info.get('phone', ''),
            extracted_info.get('aadhaar', ''),
            extracted_info.get('address', ''),
            classification_status
        )
        
    except Exception as e:
        print(f"Smart voice error: {e}")
        return f"Error: {str(e)[:50]}...", "", "", "", "", ""
    
def process_smart_voice_input(audio_file):
    if audio_file is None:
        return "No audio recorded", "", "", "", "", ""
    
    try:
        client = SarvamAI(
            api_subscription_key=SARVAM_API_KEY,
        )

        job = client.speech_to_text_translate_job.create_job(
            model="saaras:v2.5",
            num_speakers=1, 
            prompt="Administrative Complaint"
        )
        path_obj = Path(audio_file)
        job.upload_files(file_paths=[audio_file])
    
        job.start()
    
        final_status = job.wait_until_complete()
    
        if job.is_failed():
            print("STT job failed.")
            return
    
        output_dir = "./output"
        job.download_outputs(output_dir=output_dir)
        print(f"Output downloaded to: {output_dir}")
        with open(f"{output_dir}/{path_obj.name}.json", 'rb') as f:
            response = json.loads(f.read())
        print(response)
        full_text = response.get('transcript', '').strip()
        
        if not full_text.strip():
            return "Speech not recognized. Please speak clearly or type manually.", "", "", "", "", ""

        extracted_info = extract_info_from_speech(full_text)
        
        complaint_text = extracted_info.get('complaint_text', full_text)
        # print(f"Classifying complaint: {complaint_text[:100]}...")
        complaint_type = llm_categorizer.categorize_complaint(complaint_text)
        print(f"Auto-classified as: {complaint_type}")
        
        # Create classification status message
        classification_status = f"**Auto-Classified Complaint Type:** {complaint_type}"
        
        return (
            complaint_text,
            extracted_info.get('name', ''),
            extracted_info.get('phone', ''),
            extracted_info.get('aadhaar', ''),
            extracted_info.get('address', ''),
            classification_status
        )
        
    except Exception as e:
        print(f"Smart voice error: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)[:50]}...", "", "", "", "", ""


def extract_info_from_speech(text):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        extraction_prompt = f"""
Extract information from this complaint speech text in English, Hindi, or Telugu. Extract ALL available information.

Speech text: "{text}"

Return ONLY in this exact JSON format:
{{
    "name": "extracted full name or empty string",
    "phone": "9-10 digit phone number or empty string", 
    "aadhaar": "12-digit aadhaar or empty string",
    "address": "full address or empty string",
    "complaint_text": "the actual complaint description"
}}

IMPORTANT PATTERNS TO RECOGNIZE:

NAME PATTERNS:
- English: "My name is John", "I am John", "This is John"
- Hindi: "mera naam hai Rohit", "main Rohit hoon", "naam Rohit Singh"
- Telugu: "naa peru", "nenu"

PHONE NUMBER PATTERNS:
- English: "my number is 9876543210", "phone 98765 43210" (9 or 10 digits)
- Hindi: "mera number hai nau aath saat", "phone number nau aath saat cheh"
- Convert Hindi numbers: ek=1, do=2, teen=3, char=4, panch=5, cheh=6, saat=7, aath=8, nau=9, das=0
- Accept 9 or 10 digit phone numbers starting with 6, 7, 8, or 9

ADDRESS PATTERNS:
- Look for: "address", "pata", "ghar", village/mandal names, door numbers

AADHAAR PATTERNS:
- 12-digit numbers, "aadhaar", "aadhar card"

COMPLAINT PATTERNS:
- Main issue: power cut, road problem, water issue, etc.
- Hindi: "shikayat", "samasya", "pareshani"
- Telugu: "complaint", "problem"

CONVERT AND CLEAN:
- Convert Hindi number words to digits
- Extract proper names (capitalize)
- Format phone as 9-10 digits
- Keep original complaint language
"""
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        # print(f"LLM extraction result: {result_text}")
        
        try:
            extracted = json.loads(result_text)
            
            # Validate and clean extracted data
            validated_info = {
                'name': clean_name(extracted.get('name', '')),
                'phone': clean_phone(extracted.get('phone', '')),
                'aadhaar': clean_aadhaar(extracted.get('aadhaar', '')),
                'address': clean_address(extracted.get('address', '')),
                'complaint_text': extracted.get('complaint_text', text)
            }
            
            return validated_info
            
        except json.JSONDecodeError:
            print("Failed to parse LLM JSON response")
            return None
            
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return None

def clean_name(name):
    if not name:
        return ""
    name = str(name).strip().title()
    return name if re.match(r'^[A-Za-z\s]+$', name) else ""

def clean_phone(phone):
    if not phone:
        return ""
    digits = re.sub(r'\D', '', str(phone))
    if len(digits) == 9 and digits[0] in '6789':
        return digits
    elif len(digits) == 10 and digits[0] in '6789':
        return digits
    return ""

def clean_aadhaar(aadhaar):
    if not aadhaar:
        return ""
    digits = re.sub(r'\D', '', str(aadhaar))
    return digits if len(digits) == 12 else ""

def clean_address(address):
    if not address:
        return ""
    return str(address).strip()

def validate_aadhaar(aadhaar):
    return bool(re.match(r'^\d{12}$', aadhaar.replace(' ', '').replace('-', '')))

def validate_phone(phone):
    phone_digits = re.sub(r'\D', '', phone)
    return (len(phone_digits) == 9 or len(phone_digits) == 10) and phone_digits[0] in '6789'

def generate_complaint_number():
    import random
    year = datetime.datetime.now().year
    number = random.randint(100000, 999999)
    return f"AP-{year}-{number:06d}"

def load_assignment_matrix():
    return pd.read_csv("complaint_assignment_matrix_levels.csv")

def get_assignment_details(complaint_type, assignment_df):
    if complaint_type:
        matches = assignment_df[assignment_df['Complaint Type'].str.contains(complaint_type, case=False, na=False)]
        
        if not matches.empty:
            row = matches.iloc[0]
            assignee = row['Assignee']
            support_levels = [row.get('Level 1', ''), row.get('Level 2', '')]
            return assignee, support_levels
    
    return "General Administrator", ["", ""]

def calculate_sla_date(complaint_type):
    sla_days = {
        "Land mutation/name correction pending": 15,
        "Ration shop supply issues": 7,
        "MGNREGS wage problems": 10,
        "Power outages/transformer issues": 3,
        "Water supply problems": 5,
        "School MDM/infrastructure issues": 7,
        "Healthcare (PHC) issues": 5,
        "APSRTC bus services": 7,
        "Crop damage assessment request": 15,
        "Streetlights not working": 7,
        "Village road/drain repair needed": 30,
        "Others": 15  # Default 15 days for Others category
    }
    days = sla_days.get(complaint_type, 15)  # Changed default from 7 to 15
    sla_date = datetime.datetime.now() + datetime.timedelta(days=days)
    return sla_date.strftime("%Y-%m-%d"), days

def submit_complaint(name, phone, aadhaar, address, complaint_text, phone_optional=False, manual_category=None):
    errors = []
    if not name.strip():
        errors.append("Name is required")
    if phone_optional:
        if phone.strip() and not validate_phone(phone):
            errors.append("Phone number must be 9-10 digits starting with 6-9 (if provided)")
    else:
        if not validate_phone(phone):
            errors.append("Phone number must be 9-10 digits starting with 6-9")
    
    if aadhaar.strip() and not validate_aadhaar(aadhaar):
        errors.append("Aadhaar must be 12 digits if provided")
    if not complaint_text.strip():
        errors.append("Complaint description is required")
    if errors:
        return {"error": errors}
    
    try:
        if manual_category and manual_category != "Use Auto-Classified Category":
            complaint_type = manual_category
            print(f"Using manually overridden category: {complaint_type}")
        else:
            print(f"Categorizing complaint: {complaint_text[:100]}...")
            complaint_type = llm_categorizer.categorize_complaint(complaint_text)
            print(f"LLM categorized as: {complaint_type}")
        
        assignment_df = load_assignment_matrix()
        assignee, support_levels = get_assignment_details(complaint_type, assignment_df)
        print(assignee,support_levels)
        complaint_number = generate_complaint_number()
        sla_date, sla_days = calculate_sla_date(complaint_type)
        
        complaint_data = {
            "success": True,
            "complaint_number": complaint_number,
            "complaint_type": complaint_type,
            "assignee": assignee,
            "sla_date": sla_date,
            "sla_days": sla_days,
            "support_levels": support_levels,
            "name": name,
            "phone": phone,
            "aadhaar": aadhaar,
            "address": address,
            "complaint_text": complaint_text,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        tracker.add_complaint(complaint_data)
        return complaint_data
        
    except Exception as e:
        return {"error": [f"Error processing complaint: {str(e)}"]}

def display_complaint_result(result):
    if "error" in result:
        error_msg = " **Submission Failed:**\n\n"
        for error in result["error"]:
            error_msg += f"• {error}\n"
        return error_msg
    
    info = []
    info.append(f"**Complaint Successfully Registered!**")
    info.append(f"**Complaint Number:** {result['complaint_number']}")
    info.append(f"**Complaint Type:** {result['complaint_type']}")
    info.append(f"**Assigned To:** {result['assignee']}")
    info.append(f"**SLA Date:** {result['sla_date']} ({result['sla_days']} days)")
    info.append(f"\n**Complainant Details:**")
    info.append(f"- **Name:** {result['name']}")
    if result['phone']:
        info.append(f"- **Phone:** {result['phone']}")
    if result['aadhaar']:
        info.append(f"- **Aadhaar:** {'*' * 8 + result['aadhaar'][-4:]}")
    if result['address']:
        info.append(f"- **Address:** {result['address']}")
    info.append(f"\n**Next Step:** Complaint has been assigned for resolution")
    return "\n".join(info)

def check_status(search_method, search_value):
    if not search_value.strip():
        return "Please enter a valid search value"
    
    try:
        found_complaints = tracker.search_complaints(search_method, search_value)
        
        if found_complaints:
            results = []
            for complaint in found_complaints[:5]:
                result_text = f"""
 **Complaint Found:**\n
 **Complaint Number:** {complaint.get('complaint_number', 'N/A')}\n
 **Name:** {complaint.get('name', 'N/A')}\n
 **Phone:** {complaint.get('phone', 'N/A')}\n
 **Location:** {complaint.get('village', 'N/A')}, {complaint.get('mandal', 'N/A')}\n
 **Complaint Type:** {complaint.get('complaint_type', 'N/A')}\n
 **Assigned To:** {complaint.get('assignee', 'N/A')}\n
 **Status:** {complaint.get('status', 'N/A')}\n
 **Submitted:** {complaint.get('submission_time', 'N/A')}\n
 **SLA Date:** {complaint.get('sla_date', 'N/A')}\n
 **Description:** {str(complaint.get('complaint_text', ''))[:200]}...
                """
                results.append(result_text)
            return "\n".join(results)
        else:
            stats = tracker.get_complaint_stats()
            if stats['total'] == 0:
                return f" **Search Results:**\n\nNo complaints found for {search_method}: {search_value}\n\n **Note:** No complaints have been submitted yet."
            else:
                return f" **Search Results:**\n\nNo complaints found for {search_method}: {search_value}\n\n **System Stats:** {stats['total']} total complaints in database."
            
    except Exception as e:
        return f"Error searching complaints: {str(e)}\n\nPlease try again."

with gr.Blocks(title="AP Complaint Management System", css="""
    .compact-audio-upload {
        max-height: 450px !important;
    }
    .compact-audio-upload > div {
        max-height: 450px !important;
    }
    .compact-audio-upload {
        min-height: 240px !important;
        max-height: 450px !important;
    }
""") as demo:
    gr.Markdown("# AP Complaint Management System\nSubmit citizen complaints for Mandal-level resolution")
    
    # TAB 1: Voice-Based Complaint
    with gr.Tab(" Voice-Based Complaint"):
        gr.Markdown("### Smart Voice Auto-Fill - Speak once, fill everything!")
        
        with gr.Row():
            with gr.Column():
                name_voice = gr.Textbox(label="Full Name *", placeholder="Auto-filled from voice or type manually")
                phone_voice = gr.Textbox(label="Mobile Number *", placeholder="Auto-filled from voice or type manually")
                aadhaar_voice = gr.Textbox(label="Aadhaar Number (Optional)", placeholder="Auto-filled from voice or type manually")
                address_voice = gr.Textbox(label="Address (Optional)", placeholder="Auto-filled from voice or type manually", lines=2)
            
            with gr.Column():
                gr.Markdown("#### Complaint Description + Smart Voice")
                
                complaint_text_voice = gr.Textbox(
                    label="Complaint Description *", 
                    placeholder=" Click mic button OR upload WAV file OR type/paste your complaint...\n\nSupported Languages: English, Hindi, Telugu\nभाषा समर्थन: अंग्रेजी, हिंदी, तेलुगु\nభాష మద్దత: ఇంగ్లీష్, హిందీ, తెలుగు",
                    lines=4
                )
                
                process_text_btn = gr.Button(" Process Text & Auto-Fill", variant="secondary", size="sm")
                
                with gr.Row():
                    voice_input = gr.Audio(
                        label=" Record Voice", 
                        elem_classes=["compact-audio-upload"],
                        sources=["microphone"],
                        type="filepath",
                        format="wav"
                    )

                    audio_upload = gr.Audio(
                        label=" Upload WAV File",
                        sources=["upload"],
                        type="filepath",
                        format="wav"
                    )
                with gr.Row():
                    process_voice_btn = gr.Button("🎯 Process Recording & Auto-Fill", variant="primary", size="sm")
                    process_upload_btn = gr.Button("🎯 Process Upload & Auto-Fill", variant="primary", size="sm")
                
                classification_status_voice = gr.Markdown(label="Classification Status", value="")
                
        
        submit_voice_btn = gr.Button(" Submit Complaint", variant="primary", size="lg")
        output_voice = gr.Markdown()
        
        process_text_btn.click(
            process_text_input,
            inputs=[complaint_text_voice],
            outputs=[complaint_text_voice, name_voice, phone_voice, aadhaar_voice, address_voice, classification_status_voice]
        )
        
        process_voice_btn.click(
            process_smart_voice_input,
            inputs=[voice_input],
            outputs=[complaint_text_voice, name_voice, phone_voice, aadhaar_voice, address_voice, classification_status_voice]
        )
        
        process_upload_btn.click(
            process_smart_voice_input,
            inputs=[audio_upload],
            outputs=[complaint_text_voice, name_voice, phone_voice, aadhaar_voice, address_voice, classification_status_voice]
        )
        
        submit_voice_btn.click(
            lambda n, p, a, addr, c: display_complaint_result(submit_complaint(n, p, a, addr, c)),
            inputs=[name_voice, phone_voice, aadhaar_voice, address_voice, complaint_text_voice],
            outputs=output_voice
        )
    
    #TAB 2: Upload Letter/Document
    with gr.Tab(" Document Upload & OCR"):
        gr.Markdown("### Smart Document Auto-Fill - Upload once, fill everything!")
        
        with gr.Row():
            with gr.Column():
                name_doc = gr.Textbox(label="Full Name *", placeholder="Auto-filled from document or type manually")
                phone_doc = gr.Textbox(label="Mobile Number *", placeholder="Auto-filled from document or type manually")
                aadhaar_doc = gr.Textbox(label="Aadhaar Number (Optional)", placeholder="Auto-filled from document or type manually")
                address_doc = gr.Textbox(label="Address (Optional)", placeholder="Auto-filled from document or type manually", lines=2)
            
            with gr.Column():
                gr.Markdown("#### Upload Complaint Document/Letter")
                
                document_upload = gr.File(
                    label=" Upload Document Image (JPG, PNG)",
                    file_types=["image"],
                    type="filepath"
                )
                
                gr.Markdown("""
                **Supported formats:** Photos of handwritten letters, printed documents, complaint forms
                **Note:** All fields with * are mandatory
                **Auto-Classification:** Complaint will be automatically categorized and assigned
                """)
                
                process_doc_btn = gr.Button(" Process Document & Auto-Fill", variant="primary", size="lg")
                
                complaint_text_doc = gr.Textbox(
                    label="Extracted Complaint Text *",
                    lines=6,
                    placeholder="Text will be extracted and translated here automatically..."
                )
                
                classification_status_doc = gr.Markdown(label="Classification Status", value="")
        
        submit_doc_btn = gr.Button(" Submit Complaint", variant="primary", size="lg")
        output_doc = gr.Markdown()
        
        process_doc_btn.click(
            process_smart_document_input,
            inputs=[document_upload],
            outputs=[complaint_text_doc, name_doc, phone_doc, aadhaar_doc, address_doc, classification_status_doc]
        )
        
        submit_doc_btn.click(
            lambda n, p, a, addr, c, cat: display_complaint_result(submit_complaint(n, p, a, addr, c, phone_optional=False)),
            inputs=[name_doc, phone_doc, aadhaar_doc, address_doc, complaint_text_doc],
            outputs=output_doc
        )
    
    # TAB 3: Check Status
    with gr.Tab("Check Complaint Status"):
        with gr.Row():
            with gr.Column():
                search_method = gr.Dropdown(
                    choices=["Complaint Number", "Phone Number", "Aadhaar Number"],
                    label="Search By",
                    value="Phone Number"
                )
                search_value = gr.Textbox(
                    label="Enter Search Value",
                    placeholder="Enter complaint number, phone number, or Aadhaar number"
                )
                status_btn = gr.Button("Check Status", variant="primary")
        
        status_output = gr.Markdown()
        status_btn.click(check_status, inputs=[search_method, search_value], outputs=status_output)
    
    # TAB 4: Help/Information
    with gr.Tab("System Information"):
        gr.Markdown("""
        ## Smart Voice + Document OCR Auto-Fill System
        
        ### **Revolutionary Features:**
        
        #### Voice-Based Complaints:
        - ** Three Input Options**: Record, Upload WAV, or Type/Paste text
        - ** Smart Extraction**: AI extracts name, phone, address, complaint automatically
        - ** Hybrid Input**: Combine voice/text and manual typing
        - ** MacBook Mic**: Uses your built-in microphone for recording
        - ** WAV Upload**: Upload pre-recorded audio files
        - ** Text Processing**: Type or paste text for auto-extraction
        - ** Instant Processing**: Real-time speech-to-text conversion
        - ** Auto-Classification**: Complaint automatically categorized and assigned
        
        #### Document OCR Complaints:
        - ** Upload & Extract**: Take photo of handwritten/printed letter
        - ** Multi-Language OCR**: Supports English, Hindi, Telugu, Tamil, Kannada
        - ** Auto-Translation**: Converts regional languages to English
        - ** Smart Extraction**: AI extracts all details from document
        - ** Optional Phone**: Phone number not mandatory for documents
        - ** Auto-Classification**: Same intelligent categorization as voice
        
        ### **How to Use Voice Input:**
        1. **Option A - Record:** Click the microphone button and speak naturally
        2. **Option B - Upload:** Upload a pre-recorded WAV file
        3. **Option C - Type/Paste:** Type or paste text in complaint description box, then click "Process Text & Auto-Fill"
        4. **Speak/Type** including all details:
           - "My name is [Your Name]"
           - "My phone number is [9-10 digit number]"  
           - "My address is [Full Address]"
           - "My complaint is about [Describe issue]"
        5. **Click** appropriate process button (Recording/Upload/Text)
        6. **Review** auto-filled information
        7. **Submit** your complaint
        
        ### **How to Use Document Upload:**
        1. **Take a photo** or scan your complaint letter/document
        2. **Upload** the image (JPG, PNG supported)
        3. **Click** "Process Document & Auto-Fill"
        4. **Review** extracted information (edit if needed)
        5. **Submit** your complaint
        
        ### **Voice Input Example:**
        *"Hello, my name is Rajesh Kumar, my phone number is 9876543210, my Aadhaar is 123456789012, my address is D.No. 45 Gandhi Road, Guntur, Andhra Pradesh. My complaint is about street lights not working in our area for the past one week."*
        
        ### **System Features:**
        - **Smart Categorization**: Auto-identifies complaint types using AI
        - **Instant Assignment**: Maps to appropriate officials automatically
        - **SLA Management**: Provides resolution timeframes
        - **Status Tracking**: Search and monitor progress
        - **9-10 Digit Phone**: Accepts both 9 and 10 digit phone numbers
        - **Hybrid System**: Combine voice OR document upload with manual editing
        
        ### **Supported Complaint Types:**
        1. Land mutation/name correction pending
        2. Ration shop/card issues
        3. MGNREGS wage problems
        4. Power outages/transformer issues
        5. Water supply problems
        6. School MDM issues
        7. Healthcare (PHC) issues
        8. Crop damage assessment
        9. Streetlights not working
        10. Village road/drain repair
        11. Caste certificate delays
        12. And more...
        
        The system combines advanced speech recognition, OCR, and AI to make complaint submission effortless!
        """)
    
    gr.Markdown("---\n**AP Complaint Management System** | Voice + Document OCR Enabled Platform with AI Classification")

def main():
    print("Starting AP Complaint Management System (Voice + Document OCR)...")
    print("Initializing systems...")
    
    try:
        r = sr.Recognizer()
        print("Speech recognition initialized")
    except Exception as e:
        print(f"Speech recognition warning: {e}")
    
    if GEMINI_API_KEY:
        print("Gemini OCR system ready")
    else:
        print("Warning: GEMINI API KEY not found in .env file")
    
    try:
        test_complaint = "Street lights not working"
        category = llm_categorizer.categorize_complaint(test_complaint)
        print(f"LLM categorizer ready. Test result: '{category}'")
    except Exception as e:
        print(f"LLM categorizer warning: {e}")
    
    print(" Smart Voice + Document OCR system ready!")
    print(" Phone validation: 9-10 digits starting with 6-9")
    print(" Auto-classification enabled for both voice and document inputs")
    print(" Three input methods: Record, Upload WAV, Type/Paste text")
    demo.launch(share=False, server_name="127.0.0.1", server_port=8080)

if __name__ == "__main__":
    main()