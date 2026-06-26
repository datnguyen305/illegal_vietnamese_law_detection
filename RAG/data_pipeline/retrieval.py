import os
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv

def ask_traffic_law(query_text, top_k=3):
    print(f"\n[BƯỚC R - Truy xuất] Đang tìm kiếm các điều luật liên quan đến: '{query_text}'...")
    
    # 1. Load môi trường
    base_dir = os.path.dirname(__file__)
    env_path = os.path.join(base_dir, '..', '.env')
    load_dotenv(dotenv_path=env_path, override=True)

    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"Lỗi kết nối Database: {e}")
        return

    # Khởi tạo Gemini Client
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    # 2. NHÚNG CÂU HỎI (Dùng model Embedding)
    response_embed = client.models.embed_content(
        model='gemini-embedding-001',
        contents=query_text,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    query_vector = response_embed.embeddings[0].values

    # 3. TRUY VẤN VECTOR TỪ POSTGRESQL
    search_query = """
        SELECT dieu, hanh_vi_vi_pham, 1 - (embedding <=> %s::vector) AS similarity
        FROM luat_giao_thong
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    cursor.execute(search_query, (query_vector, query_vector, top_k))
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()

    if not results:
        print("Không tìm thấy dữ liệu luật nào trong Database.")
        return

    # 4. ĐÓNG GÓI NGỮ CẢNH (Context Enrichment)
    print("📦 Đang gom các điều luật thô thành 'phao cứu sinh' cho AI...")
    context_text = ""
    for i, row in enumerate(results):
        dieu = row[0]
        noi_dung = row[1]
        context_text += f"\n--- CĂN CỨ SỐ {i+1} ---\n📍 Vị trí: {dieu}\n📜 Nội dung: {noi_dung}\n"

    # 5. BƯỚC G - SINH VĂN BẢN (Dùng model LLM)
    print(f"🤖 [BƯỚC G - Tổng hợp] AI đang soạn biên bản phạt nguội...\n")
    
    # Kỹ thuật Prompt Engineering: Ra lệnh cho AI đóng vai và ép khuôn câu trả lời
    prompt = f"""Bạn là một hệ thống AI hỗ trợ Cảnh sát Giao thông đường bộ Việt Nam. 
Nhiệm vụ của bạn là đọc các Căn cứ Pháp lý dưới đây và trả lời chính xác hành vi vi phạm.

QUY TẮC NGHIÊM NGẶT:
1. TUYỆT ĐỐI không tự bịa ra luật. CHỈ sử dụng thông tin trong phần [CĂN CỨ PHÁP LÝ] được cung cấp. Nếu trong căn cứ không có thông tin, hãy nói "Chưa đủ dữ liệu để kết luận".
2. Trình bày giống như một thông báo phạt nguội: Ngắn gọn, có gạch đầu dòng rõ ràng.
3. Bắt buộc phải nêu rõ: Mức phạt tiền (từ bao nhiêu đến bao nhiêu) và Các hình phạt bổ sung (nếu có, ví dụ: tước bằng).
4. Phải trích dẫn rõ tên Điều/Khoản được áp dụng.

[CĂN CỨ PHÁP LÝ TỪ DATABASE]
{context_text}

[HÀNH VI PHÁT HIỆN TỪ CAMERA]
{query_text}

[KẾT LUẬN XỬ PHẠT CỦA BẠN]
"""    
    response_llm = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    print("="*70)
    print("KẾT QUẢ TỪ HỆ THỐNG RAG")
    print("="*70)
    print(response_llm.text)
    print("="*70)

if __name__ == '__main__':
    # Giả lập đầu vào từ Camera của bạn đồng đội
    hanh_vi_camera_phat_hien = "xe máy chạy vượt đèn đỏ"
    
    ask_traffic_law(hanh_vi_camera_phat_hien, top_k=3)