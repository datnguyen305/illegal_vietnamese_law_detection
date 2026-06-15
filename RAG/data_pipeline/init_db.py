import os
import psycopg2
from dotenv import load_dotenv

# Load biến môi trường từ file .env ở thư mục cha
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def init_database():
    print("Đang kết nối đến PostgreSQL...")
    try:
        # Kết nối DB
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        conn.autocommit = True # Để tự động commit sau mỗi lệnh
        cursor = conn.cursor()

        # 1. Bật extension pgvector
        print("Bật extension pgvector...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. Xóa bảng cũ nếu tồn tại (để dễ test lại nhiều lần)
        cursor.execute("DROP TABLE IF EXISTS luat_giao_thong;")

        # 3. Tạo bảng lưu trữ Chunk luật và Vector (768 chiều cho model tiếng Việt)
        print("Khởi tạo bảng luat_giao_thong...")
        create_table_query = """
        CREATE TABLE luat_giao_thong (
            id VARCHAR(100) PRIMARY KEY,
            van_ban_goc VARCHAR(50),
            chuong VARCHAR(255),
            dieu VARCHAR(255),
            khoan_diem VARCHAR(255),
            hanh_vi_vi_pham TEXT,
            metadata JSONB,
            embedding VECTOR(768)
        );
        """
        cursor.execute(create_table_query)
        
        # 4. Tạo Index cho vector để search siêu tốc (IVFFlat)
        cursor.execute("CREATE INDEX ON luat_giao_thong USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")

        print("Khởi tạo Database và Bảng Vector thành công!")
        
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    init_database()