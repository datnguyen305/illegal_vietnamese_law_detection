import os
import json
import psycopg2
from google import genai
import time
from google.genai import types
from dotenv import load_dotenv

def process_silver_to_gold():
    print("Bắt đầu tiến trình Nhúng Vector (Chế độ Nạp Tăng Cường)...")
    
    base_dir = os.path.dirname(__file__)
    env_path = os.path.join(base_dir, '..', '.env')
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    silver_path = os.path.join(base_dir, '..', 'data', 'silver_lake', 'silver_168_2024.json')
    with open(silver_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()

    # --- TÍNH NĂNG MỚI: LẤY DANH SÁCH ID ĐÃ NHÚNG ---
    cursor.execute("SELECT id FROM luat_giao_thong;")
    existing_ids = {row[0] for row in cursor.fetchall()}
    print(f"📦 Đã tìm thấy {len(existing_ids)} chunks trong Database. Sẽ bỏ qua các chunk này!")

    chunks_to_process = [c for c in chunks if c["id"] not in existing_ids]
    print(f"Chỉ còn {len(chunks_to_process)} chunks cần xử lý...")
    
    if len(chunks_to_process) == 0:
        print("Mọi dữ liệu đã được nạp đầy đủ. Không cần chạy thêm!")
        return

    success_count = 0
    for i, chunk in enumerate(chunks_to_process):
        text_to_embed = chunk["hanh_vi_vi_pham"]
        try:
            # Gọi API
            response = client.models.embed_content(
                model='gemini-embedding-001',
                contents=text_to_embed,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            embedding_vector = response.embeddings[0].values
            
            # Lưu DB
            insert_query = """
                INSERT INTO luat_giao_thong (id, van_ban_goc, chuong, dieu, khoan_diem, hanh_vi_vi_pham, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            metadata_json = json.dumps(chunk["metadata"], ensure_ascii=False)
            
            cursor.execute(insert_query, (
                chunk["id"], chunk["van_ban_goc"], chunk["chuong"], chunk["dieu"],
                chunk.get("khoan_diem", ""), text_to_embed, metadata_json, embedding_vector
            ))
            
            success_count += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks_to_process):
                print(f"⏳ Đã nạp {success_count}/{len(chunks_to_process)} chunks mới vào Database...")
                conn.commit() # Commit ngay từng đợt nhỏ để an toàn
                
        except Exception as e:
                    print(f"Lỗi tại chunk {chunk['id']}: {e}")
                    conn.rollback() 
                    if "429" in str(e):
                        print("Bị Google tuýt còi vì quá tốc độ (100 req/phút). Đang tự động ngủ 60 giây để chờ hồi Quota...")
                        time.sleep(60) # Tự động ngủ 1 phút
                        print("Đã hồi sức! Chuẩn bị chạy tiếp...")
                        # Tiếp tục vòng lặp
                        continue
                    continue

    cursor.close()
    conn.close()
    print(f"HOÀN TẤT ĐỢT NÀY! Đã nạp thêm {success_count} vector.")

if __name__ == '__main__':
    process_silver_to_gold()