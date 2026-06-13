import os
import json
import re
from bs4 import BeautifulSoup

def process_bronze_to_silver():
    # 1. Thiết lập đường dẫn thư mục
    base_dir = os.path.dirname(__file__)
    bronze_path = os.path.join(base_dir, '..', 'data', 'bronze_lake', 'bronze_vbpl_168_2024.html')
    
    silver_dir = os.path.join(base_dir, '..', 'data', 'silver_lake')
    os.makedirs(silver_dir, exist_ok=True)
    silver_path = os.path.join(silver_dir, 'silver_168_2024.json')

    print("🔄 Bắt đầu bóc tách dữ liệu từ Tầng Bronze sang Tầng Silver...")

    # 2. Đọc file HTML thô
    try:
        with open(bronze_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file tại: {bronze_path}. Hãy kiểm tra lại tên file.")
        return

    chunks = []
    
    # Biến nhớ ngữ cảnh hiện tại
    current_chapter = "Chưa xác định Chương"
    current_article = "Chưa xác định Điều"

    # 3. Quét tất cả các thẻ có chứa luật (dựa trên class CSS đã phân tích)
    # Dùng Regex để tóm gọn tất cả các class bắt đầu bằng "prov-"
    elements = soup.find_all(class_=re.compile(r'prov-(chapter|article|clause|item)'))

    for el in elements:
        # Lấy text, bỏ khoảng trắng thừa
        text = el.get_text(separator=" ", strip=True)
        if not text:
            continue
            
        classes = el.get('class', [])

        if 'prov-chapter' in classes:
            current_chapter = text # Cập nhật Chương đang đọc
            
        elif 'prov-article' in classes:
            current_article = text # Cập nhật Điều đang đọc
            
        elif 'prov-clause' in classes or 'prov-item' in classes:
            # Gặp Khoản hoặc Điểm -> Đây chính là nội dung lỗi vi phạm cần băm nhỏ!
            
            # Kỹ thuật Context Enrichment: Nối ngữ cảnh vào đầu câu
            enriched_text = f"[Văn bản: Nghị định 168/2024/NĐ-CP] - [{current_chapter}] - [{current_article}] - Nội dung: {text}"
            
            chunk_id = f"ND168_chunk_{len(chunks) + 1}"
            
            # Đóng gói thành cấu trúc JSON chuẩn bị cho Database
            chunk_data = {
                "id": chunk_id,
                "van_ban_goc": "Nghị định 168/2024/NĐ-CP",
                "chuong": current_chapter,
                "dieu": current_article,
                "hanh_vi_vi_pham": enriched_text,
                "metadata": {
                    "nguon": "vbpl.vn",
                    "loai_the": "khoan_diem"
                }
            }
            chunks.append(chunk_data)

    # 4. Lưu kết quả ra file JSON ở Silver Lake
    with open(silver_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    print(f"✅ Bóc tách thành công! Đã tạo ra {len(chunks)} Chunks luật.")
    print(f"📁 Dữ liệu sạch được lưu tại: {silver_path}")

if __name__ == '__main__':
    process_bronze_to_silver()