import pandas as pd
from sqlalchemy import create_engine

# ฟังก์ชันในการดึงข้อมูลจากฐานข้อมูลและบันทึกเป็น CSV
def save_data_to_csv():
    try:
        # สร้างการเชื่อมต่อกับฐานข้อมูล
        engine = create_engine('mysql+mysqlconnector://root:1234@localhost/reviewapp')
        
        # Query สำหรับข้อมูล Content-Based
        query_content = "SELECT * FROM contentbasedview;"
        content_based_data = pd.read_sql(query_content, con=engine)
        print("โหลดข้อมูล Content-Based สำเร็จ")

        # Query สำหรับข้อมูล Collaborative
        query_collaborative = "SELECT * FROM collaborativeview;"
        collaborative_data = pd.read_sql(query_collaborative, con=engine)
        print("โหลดข้อมูล Collaborative สำเร็จ")
        
        # บันทึกข้อมูลลงไฟล์ CSV
        content_based_data.to_csv('contentbasedview.csv', index=False)
        collaborative_data.to_csv('collaborativeview.csv', index=False)
        print("บันทึกข้อมูลลงไฟล์ CSV สำเร็จ")
    
    except Exception as e:
        print(f"ข้อผิดพลาดในการดึงข้อมูลและบันทึกเป็น CSV: {str(e)}")
        raise

# เรียกใช้งานฟังก์ชัน
save_data_to_csv()
