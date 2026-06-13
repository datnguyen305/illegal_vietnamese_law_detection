import os
import psycopg2
from dotenv import load_dotenv

# Load môi trường
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def fix_schema():
    print("🛠️ Đang sửa lỗi cấu trúc bảng luat_giao_thong...")
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Nới rộng các cột VARCHAR(255) thành TEXT
        alter_query = """
        ALTER TABLE luat_giao_thong 
        ALTER COLUMN chuong TYPE TEXT,
        ALTER COLUMN dieu TYPE TEXT,
        ALTER COLUMN khoan_diem TYPE TEXT,
        ALTER COLUMN van_ban_goc TYPE TEXT;
        """
        cursor.execute(alter_query)
        print("✅ Đã nới rộng các cột thành kiểu TEXT thành công! Không mất dữ liệu cũ.")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    fix_schema()