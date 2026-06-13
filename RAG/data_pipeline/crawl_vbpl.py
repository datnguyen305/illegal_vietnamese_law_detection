import os
import json
from datetime import datetime
from scrapling import Fetcher

def crawl_vbpl_direct(so_hieu, direct_url):
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'bronze_lake')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Bắt đầu cào dữ liệu cho: {so_hieu}")
    fetcher = Fetcher()
    
    try:
        print(f"👉 Truy cập trực tiếp trang chi tiết: {direct_url}")
        detail_page = fetcher.get(direct_url)
        
        # --- BƯỚC DEBUG: In thử 500 ký tự đầu tiên web trả về ---
        print("\n--- [DEBUG log] Web trả về nội dung như sau: ---")
        print(detail_page.text[:500] if detail_page.text else "Trang web trả về rỗng hoàn toàn!")
        print("--------------------------------------------------\n")
        
        # Mở rộng lưới bắt HTML bằng nhiều bộ CSS Selector kết hợp
        raw_html_content = detail_page.css('#toanvancontent, .toanvancontent, .content, .document-content, article, main').extract_first()
        
        # Nếu vẫn không bắt được, bốc toàn bộ thẻ <body>
        if not raw_html_content:
             print("⚠️ Không tìm thấy khung chứa luật cụ thể, đang cào toàn bộ thẻ <body>...")
             raw_html_content = detail_page.css('body').extract_first()
             
        bronze_data = {
            "document_id": so_hieu,
            "source": "vbpl.vn",
            "crawl_timestamp": datetime.now().isoformat(),
            "metadata": {
                "trang_thai": "Còn hiệu lực", 
                "ngay_ban_hanh": "26/12/2024",
                "ngay_hieu_luc": "01/01/2025",
                "detail_url": direct_url
            },
            "raw_html": raw_html_content if raw_html_content else "Lỗi bóc tách nội dung"
        }
        
        file_name = f"bronze_vbpl_{so_hieu.replace('/', '_')}.json"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(bronze_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Đã lưu file HTML thô tại: {file_path}")

    except Exception as e:
        print(f"❌ Xảy ra lỗi: {e}")

if __name__ == "__main__":
    LINK_CHI_TIET = "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-168-2024-nd-cp-quy-dinh-xu-phat-vi-pham-hanh-chinh-ve-trat-tu-an-toan-giao-thong-trong-linh-vuc-giao-thong-duong-bo-tru-diem-phuc-hoi-diem-giay-phep-lai-xe--173920" 
    
    crawl_vbpl_direct("168/2024/NĐ-CP", LINK_CHI_TIET)