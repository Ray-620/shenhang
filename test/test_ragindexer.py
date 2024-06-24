import sys,os
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))
#sys.path.append("/home/ubuntu/Flask/ai/")
import unittest
import pandas as pd
from loguru import logger
import faiss
from ai.match import compare,answer_generative_answer
from app import create_single_index_question
import json


def deserialize_faiss_index_from_efs(efs_mount_point, file_name):
    """
    Read a Faiss index from Amazon EFS and deserialize it.

    :param efs_mount_point: EFS file system mount point path.
    :param file_name: Name of the file to read.
    :return: Faiss index object or None if reading fails.
    """
    try:
        # Specify the file path to read from
        file_path = os.path.join(efs_mount_point, file_name)
        # Deserialize the Faiss index
        index = faiss.read_index(file_path)
        return index
    except Exception as e:
        print(f"Error reading Faiss index: {str(e)}")
        return None

class testCsvIndexBuilder(unittest.TestCase):
    
    def test_create_single_index_question(self):
        """
        Scenario: Compare index file
        Args:
        """
        knowledge_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/embedding.parquet")
        knowledge_data = pd.read_parquet(knowledge_data_path)
        knowledge_index = create_single_index_question(knowledge_data)
        # test index
        standard_knowledge_index_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data")
        standard_knowledge_index = deserialize_faiss_index_from_efs(standard_knowledge_index_path,"standard_knowledge_index")
        self.assertEqual(knowledge_index.ntotal, standard_knowledge_index.ntotal)
        self.assertEqual(knowledge_index.d, standard_knowledge_index.d)
        
    def test_compare(self):
        """
        Scenario: Test retrieve
        Args:
        """
        knowledge_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/embedding.parquet")
        knowledge_data = pd.read_parquet(knowledge_data_path)
        knowledge_index = create_single_index_question(knowledge_data)
        question = "What is the born date of John Smith?"
        retreive_results = compare(question,knowledge_index,knowledge_data)
        #
        standard_results = [{'question': 'What is the born date of John Smith?',
                            'id': '83',
                            'row_as_text': '{"Client UID in AMS": "Repair9916", "Business Client Name": "Redmond Werkshop", "Industry Category": "Automobile Repair", "Contact Person Full Name": "Emily Johnson", "Date of Birth": "9/3/1988", "Gender": "Female", "Phone Number": "(425) 885-7677", "Email Address": "emily.johnson@redmondwerkshop.com", "Mailing Address": "15205 NE 90th St", "City": "Redmond", "State": "Washington", "Postal Code": 98052.0, "Claim ID": "CLM2058", "Claim Date": "4/2/2024", "Claim Type": "Wheel Balancing", "Claim Status": "Pending", "Claim Amount": "$150 ", "Occupation": "Auto Electrician", "Residence Type": "Condo", "Driving Record": "Clean", "Credit Score": 710.0, "Property Details": "Garage", "Underwriting Notes": "Requires Review", "Agent Comments": NaN, "BOP_ Policy Number": "AROP016", "BOP_ Policy Type": "Standard", "BOP_ Effective Date": "4/1/2025", "BOP_ Expiration Date": "4/1/2026", "BOP_ Premium Amount": "$5,500 ", "BOP_ Coverage Limits": "$1,000,000 ", "BOP_ General Liability": "Included", "BOP_ Property": "Included", "BOP_ Business Income": "Included", "BOP_ Equipment Coverage": "$60,000 ", "BOP_ Cyber Liability": "$35,000 ", "BOP_ Professional Liability": "$120,000 ", "CL_Auto_ Policy Number": "CACP014", "CL_Auto_ Policy Type": "Enhanced", "CL_Auto_ Effective Date": "2/18/2025", "CL_Auto_ Expiration Date": "2/18/2026", "CL_Auto_ Premium Amount": "$9,800 ", "CL_Auto_ Coverage Limits": "$1,700,000 ", "CL_Auto_ Vehicle": "Tanker Truck", "CL_Auto_ Accident and Damage": "Included", "CL_Auto_ Medical Payments": "$45,000 ", "CL_Auto_ Uninsured/Underinsured Motorist": "$180,000 ", "CL_Prop_ Policy Number": "ARCL016", "CL_Prop_ Policy Type": "Basic", "CL_Prop_ Effective Date": "4/14/2025", "CL_Prop_ Expiration Date": "4/14/2026", "CL_Prop_ Premium Amount": "$4,200 ", "CL_Prop_ Coverage Limits": "$800,000 ", "CL_Prop_ Property": "Auto Service Center", "CL_Prop_ Coverage Types": "Building", "CL_Prop_ Perils Covered": "Fire, Vandalism", "CL_Prop_ Business Interruption": "Included", "CL_Prop_ Boiler and Machinery": "$25,000 ", "id": 83}',
                            'indexed_row_as_text': '{"Business Client Name": "Redmond Werkshop", "Industry Category": "Automobile Repair", "Contact Person Full Name": "Emily Johnson", "Date of Birth": "9/3/1988", "Gender": "Female", "Phone Number": "(425) 885-7677", "Email Address": "emily.johnson@redmondwerkshop.com", "Mailing Address": "15205 NE 90th St", "City": "Redmond", "State": "Washington", "Postal Code": 98052.0}',
                            'score': 73.8,
                            'algorithm': 'FAISS'},
                            {'question': 'What is the born date of John Smith?',
                            'id': '74',
                            'row_as_text': '{"Client UID in AMS": "Repair9907", "Business Client Name": "Knight Claremont Chrysler Dodge Jeep Ram", "Industry Category": "Automobile Repair", "Contact Person Full Name": "Daniel Smith", "Date of Birth": "7/8/1972", "Gender": "Male", "Phone Number": "(909) 962-7557", "Email Address": "daniel.smith@knightclaremont.com", "Mailing Address": "620 Auto Center Dr", "City": "Claremont", "State": "California", "Postal Code": 91711.0, "Claim ID": "CLM2049", "Claim Date": "2/15/2024", "Claim Type": "Tire Replacement", "Claim Status": "Approved", "Claim Amount": "$400 ", "Occupation": "Automotive Engineer", "Residence Type": "House", "Driving Record": "Clean", "Credit Score": 780.0, "Property Details": "Garage", "Underwriting Notes": NaN, "Agent Comments": NaN, "BOP_ Policy Number": "AROP007", "BOP_ Policy Type": "Standard", "BOP_ Effective Date": "7/8/2024", "BOP_ Expiration Date": "7/8/2025", "BOP_ Premium Amount": "$4,200 ", "BOP_ Coverage Limits": "$600,000 ", "BOP_ General Liability": "Included", "BOP_ Property": "Included", "BOP_ Business Income": "Included", "BOP_ Equipment Coverage": "$35,000 ", "BOP_ Cyber Liability": "$18,000 ", "BOP_ Professional Liability": "$70,000 ", "CL_Auto_ Policy Number": "CACP005", "CL_Auto_ Policy Type": "Enhanced", "CL_Auto_ Effective Date": "5/20/2024", "CL_Auto_ Expiration Date": "5/20/2025", "CL_Auto_ Premium Amount": "$7,200 ", "CL_Auto_ Coverage Limits": "$1,200,000 ", "CL_Auto_ Vehicle": "Pickup Truck", "CL_Auto_ Accident and Damage": "Included", "CL_Auto_ Medical Payments": "$30,000 ", "CL_Auto_ Uninsured/Underinsured Motorist": "$120,000 ", "CL_Prop_ Policy Number": "ARCL007", "CL_Prop_ Policy Type": "Standard", "CL_Prop_ Effective Date": "7/8/2024", "CL_Prop_ Expiration Date": "7/8/2025", "CL_Prop_ Premium Amount": "$5,200 ", "CL_Prop_ Coverage Limits": "$900,000 ", "CL_Prop_ Property": "Transmission Shop", "CL_Prop_ Coverage Types": "Building", "CL_Prop_ Perils Covered": "Fire, Water Damage", "CL_Prop_ Business Interruption": "Included", "CL_Prop_ Boiler and Machinery": "$30,000 ", "id": 74}',
                            'indexed_row_as_text': '{"Business Client Name": "Knight Claremont Chrysler Dodge Jeep Ram", "Industry Category": "Automobile Repair", "Contact Person Full Name": "Daniel Smith", "Date of Birth": "7/8/1972", "Gender": "Male", "Phone Number": "(909) 962-7557", "Email Address": "daniel.smith@knightclaremont.com", "Mailing Address": "620 Auto Center Dr", "City": "Claremont", "State": "California", "Postal Code": 91711.0}',
                            'score': 73.6,
                            'algorithm': 'FAISS'},
                            {'question': 'What is the born date of John Smith?',
                            'id': '0',
                            'row_as_text': '{"Client UID in AMS": "SAMPLE1", "Business Client Name": "Allied Transport Logistics", "Industry Category": "Transportation", "Contact Person Full Name": "John Smith", "Date of Birth": "5/15/1975", "Gender": "Male", "Phone Number": "123-456-7890", "Email Address": "john@example.com", "Mailing Address": "1600B SW Dash Point Rd #205", "City": "Federal Way", "State": "WA", "Postal Code": 98023.0, "Claim ID": "CLM2007", "Claim Date": "2/14/2023", "Claim Type": "Theft", "Claim Status": "Open", "Claim Amount": "12500", "Occupation": "Manager", "Residence Type": "Residential", "Driving Record": "No accidents", "Credit Score": 670.0, "Property Details": "Suburban, 1400 sq.ft", "Underwriting Notes": "Moderate risk area", "Agent Comments": "Awaiting review", "BOP_ Policy Number": "CL777770", "BOP_ Policy Type": "Standard", "BOP_ Effective Date": "6/1/2023", "BOP_ Expiration Date": "5/31/2024", "BOP_ Premium Amount": "$1,190 ", "BOP_ Coverage Limits": "$4,000,000 ", "BOP_ General Liability": "Included", "BOP_ Property": "$2,000,000 ", "BOP_ Business Income": "$25,000 ", "BOP_ Equipment Coverage": "$37,578 ", "BOP_ Cyber Liability": "Included", "BOP_ Professional Liability": "Not applicable", "CL_Auto_ Policy Number": NaN, "CL_Auto_ Policy Type": NaN, "CL_Auto_ Effective Date": NaN, "CL_Auto_ Expiration Date": NaN, "CL_Auto_ Premium Amount": NaN, "CL_Auto_ Coverage Limits": NaN, "CL_Auto_ Vehicle": NaN, "CL_Auto_ Accident and Damage": NaN, "CL_Auto_ Medical Payments": NaN, "CL_Auto_ Uninsured/Underinsured Motorist": NaN, "CL_Prop_ Policy Number": NaN, "CL_Prop_ Policy Type": NaN, "CL_Prop_ Effective Date": NaN, "CL_Prop_ Expiration Date": NaN, "CL_Prop_ Premium Amount": NaN, "CL_Prop_ Coverage Limits": NaN, "CL_Prop_ Property": NaN, "CL_Prop_ Coverage Types": NaN, "CL_Prop_ Perils Covered": NaN, "CL_Prop_ Business Interruption": NaN, "CL_Prop_ Boiler and Machinery": NaN, "id": 0}',
                            'indexed_row_as_text': '{"Business Client Name": "Allied Transport Logistics", "Industry Category": "Transportation", "Contact Person Full Name": "John Smith", "Date of Birth": "5/15/1975", "Gender": "Male", "Phone Number": "123-456-7890", "Email Address": "john@example.com", "Mailing Address": "1600B SW Dash Point Rd #205", "City": "Federal Way", "State": "WA", "Postal Code": 98023.0}',
                            'score': 73.5,
                            'algorithm': 'FAISS'},
                            {'question': 'What is the born date of John Smith?',
                            'id': '41',
                            'row_as_text': '{"Client UID in AMS": "FOOD6635", "Business Client Name": "The Inn at Little Washington", "Industry Category": "Restaurant", "Contact Person Full Name": "Patrick O\\ufffd\\ufffdConnell", "Date of Birth": "3/28/1955", "Gender": "Male", "Phone Number": "(540) 123-4567", "Email Address": "patrick@theinnatlittlewashington.com", "Mailing Address": "309 Middle St", "City": "Washington", "State": "VA", "Postal Code": 22747.0, "Claim ID": "CLM2004", "Claim Date": "7/5/2022", "Claim Type": "Vandalism", "Claim Status": "Closed", "Claim Amount": "5000", "Occupation": "Waitstaff", "Residence Type": "Residential", "Driving Record": "No claims", "Credit Score": 690.0, "Property Details": "Suburban, 1200 sq.ft", "Underwriting Notes": "Moderate risk area", "Agent Comments": "Settled", "BOP_ Policy Number": "FBOP1035", "BOP_ Policy Type": "Basic", "BOP_ Effective Date": "12/10/2024", "BOP_ Expiration Date": "12/9/2025", "BOP_ Premium Amount": "$4,800 ", "BOP_ Coverage Limits": "$900,000 ", "BOP_ General Liability": "Included", "BOP_ Property": "$140,000 ", "BOP_ Business Income": "$35,000 ", "BOP_ Equipment Coverage": "$7,000 ", "BOP_ Cyber Liability": "Included", "BOP_ Professional Liability": "Not applicable", "CL_Auto_ Policy Number": "CLAUTO1035", "CL_Auto_ Policy Type": "Basic", "CL_Auto_ Effective Date": "4/10/2024", "CL_Auto_ Expiration Date": "4/9/2025", "CL_Auto_ Premium Amount": "$4,200 ", "CL_Auto_ Coverage Limits": "$800,000 ", "CL_Auto_ Vehicle": "Truck", "CL_Auto_ Accident and Damage": "Included", "CL_Auto_ Medical Payments": "$8,000 ", "CL_Auto_ Uninsured/Underinsured Motorist": "Included", "CL_Prop_ Policy Number": "CLPROP2035", "CL_Prop_ Policy Type": "Basic", "CL_Prop_ Effective Date": "4/10/2024", "CL_Prop_ Expiration Date": "4/9/2025", "CL_Prop_ Premium Amount": "$6,500 ", "CL_Prop_ Coverage Limits": "$1,500,000 ", "CL_Prop_ Property": "Building C", "CL_Prop_ Coverage Types": "Fire, Water", "CL_Prop_ Perils Covered": "Limited perils", "CL_Prop_ Business Interruption": "Included", "CL_Prop_ Boiler and Machinery": "Not applicable", "id": 41}',
                            'indexed_row_as_text': '{"Business Client Name": "The Inn at Little Washington", "Industry Category": "Restaurant", "Contact Person Full Name": "Patrick O\\ufffd\\ufffdConnell", "Date of Birth": "3/28/1955", "Gender": "Male", "Phone Number": "(540) 123-4567", "Email Address": "patrick@theinnatlittlewashington.com", "Mailing Address": "309 Middle St", "City": "Washington", "State": "VA", "Postal Code": 22747.0}',
                            'score': 73.4,
                            'algorithm': 'FAISS'},
                            {'question': 'What is the born date of John Smith?',
                            'id': '6',
                            'row_as_text': '{"Client UID in AMS": "SAMPLE7", "Business Client Name": "The Rainbow School", "Industry Category": "Education", "Contact Person Full Name": "David Johnson", "Date of Birth": "2/9/1985", "Gender": "Male", "Phone Number": "617-333-4444", "Email Address": "david@example.com", "Mailing Address": "25620 SE 39th Way", "City": "Issaquah", "State": "WA", "Postal Code": 98029.0, "Claim ID": NaN, "Claim Date": NaN, "Claim Type": NaN, "Claim Status": NaN, "Claim Amount": NaN, "Occupation": NaN, "Residence Type": NaN, "Driving Record": NaN, "Credit Score": NaN, "Property Details": NaN, "Underwriting Notes": NaN, "Agent Comments": NaN, "BOP_ Policy Number": "CL987654", "BOP_ Policy Type": "Standard", "BOP_ Effective Date": "3/5/2023", "BOP_ Expiration Date": "3/4/2024", "BOP_ Premium Amount": "$1,190 ", "BOP_ Coverage Limits": "$4,000,000 ", "BOP_ General Liability": "Included", "BOP_ Property": "$2,000,000 ", "BOP_ Business Income": "$25,000 ", "BOP_ Equipment Coverage": "$37,578 ", "BOP_ Cyber Liability": "Included", "BOP_ Professional Liability": "Not applicable", "CL_Auto_ Policy Number": NaN, "CL_Auto_ Policy Type": NaN, "CL_Auto_ Effective Date": NaN, "CL_Auto_ Expiration Date": NaN, "CL_Auto_ Premium Amount": NaN, "CL_Auto_ Coverage Limits": NaN, "CL_Auto_ Vehicle": NaN, "CL_Auto_ Accident and Damage": NaN, "CL_Auto_ Medical Payments": NaN, "CL_Auto_ Uninsured/Underinsured Motorist": NaN, "CL_Prop_ Policy Number": NaN, "CL_Prop_ Policy Type": NaN, "CL_Prop_ Effective Date": NaN, "CL_Prop_ Expiration Date": NaN, "CL_Prop_ Premium Amount": NaN, "CL_Prop_ Coverage Limits": NaN, "CL_Prop_ Property": NaN, "CL_Prop_ Coverage Types": NaN, "CL_Prop_ Perils Covered": NaN, "CL_Prop_ Business Interruption": NaN, "CL_Prop_ Boiler and Machinery": NaN, "id": 6}',
                            'indexed_row_as_text': '{"Business Client Name": "The Rainbow School", "Industry Category": "Education", "Contact Person Full Name": "David Johnson", "Date of Birth": "2/9/1985", "Gender": "Male", "Phone Number": "617-333-4444", "Email Address": "david@example.com", "Mailing Address": "25620 SE 39th Way", "City": "Issaquah", "State": "WA", "Postal Code": 98029.0}',
                            'score': 73.3,
                            'algorithm': 'FAISS'}]
        self.assertEqual(retreive_results,standard_results)
    
    # def test_answer_generative_answer(self):
    #     """
    #     Scenario: Test LLM. Since the response from GPT is random, so we can't directly compare the response result
    #     Args:
    #     """
    #     question = "What is the born date of John Smith?"
    #     knowledge_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/embedding.parquet")
    #     knowledge_data = pd.read_parquet(knowledge_data_path)
    #     knowledge_index = create_single_index_question(knowledge_data)
    #     generative_answer = answer_generative_answer(question,knowledge_index,knowledge_data,top_k_hits=5)
    #     # standard
    #     standard_generative_answer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/rag_response_example.json")
    #     with open(standard_generative_answer_path,"r") as f:
    #         standard_generative_answer = json.load(f)
    #     self.assertEqual(generative_answer,standard_generative_answer)

if __name__ == '__main__':
    unittest.main()