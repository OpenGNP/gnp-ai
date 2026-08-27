# เทียบ embedding model สำหรับ topic clustering

รันเมื่อ: บันทึกครั้งนี้ | ข้อมูล: 145 atomic points จาก 111 responses (เฉลย)
คำสั่งที่ใช้: `python3 scripts/cluster_points.py --embedding-model <model> --examples 5`

ไฟล์ผลเต็ม (ไม่ commit เพราะมี feedback จริง): `data/cluster_report_L6.txt`,
`data/cluster_report_L12.txt`, `data/cluster_report_mpnet.txt`

## สรุปทุกค่า mcs

| model | mcs | topics | outlier | npmi | diversity |
|---|---|---|---|---|---|
| **MiniLM-L6** (384d, ปัจจุบัน) | 2 | 17 | 11.0% | +0.112 | 0.99 |
| | 3 | 12 | 20.7% | **+0.156** | 0.98 |
| | 4 | 8 | 21.4% | -0.085 | 0.96 |
| | 5 | 8 | 24.8% | -0.097 | 0.99 |
| | 6 | 5 | 12.4% | -0.185 | 1.00 |
| | 7 | 5 | 15.2% | -0.215 | 0.98 |
| | 8 | 4 | 9.0% | -0.297 | 1.00 |
| | 10 | 3 | 20.0% | -0.056 | 0.97 |
| **MiniLM-L12** (384d) | 2 | 20 | 14.5% | **+0.133** | 0.99 |
| | 3 | 10 | 17.2% | -0.152 | 0.99 |
| | 4 | 6 | 9.0% | -0.249 | 1.00 |
| | 5 | 5 | 6.9% | -0.253 | 0.98 |
| | 6 | 3 | 1.4% | -0.389 | 1.00 |
| | 7 | 3 | 1.4% | -0.389 | 1.00 |
| | 8 | 3 | 1.4% | -0.422 | 1.00 |
| | 10 | 2 | 6.9% | -0.100 | 1.00 |
| **mpnet-base** (768d) | 2 | 17 | 8.3% | **+0.085** | 0.98 |
| | 3 | 11 | 13.1% | +0.011 | 0.98 |
| | 4 | 10 | 18.6% | -0.008 | 0.99 |
| | 5 | 8 | 24.8% | -0.155 | 1.00 |
| | 6 | 6 | 24.1% | -0.234 | 1.00 |
| | 7 | 5 | 26.9% | -0.339 | 1.00 |
| | 8 | 4 | 31.7% | -0.110 | 1.00 |
| | 10 | 3 | 25.5% | -0.083 | 1.00 |

## เทียบชื่อ topic ที่ mcs=5 (ค่าที่แนะนำไว้ก่อนหน้าสำหรับ MiniLM-L6)

**MiniLM-L6** — 8 topics แยกจากกันชัด reach กลาง 10 คน
```
international, activities, thai            22 จุด / 17 คน
staff, nick, pdj                            17 จุด / 16 คน
courses, curriculum, data science           26 จุด / 15 คน  (เฟ้อ 1.73x)
wifi, cb2, laptops                          11 จุด / 11 คน
lx building, cb, ac                         10 จุด /  9 คน
good, clean (คำชม)                           9 จุด /  9 คน
process, vm, request                         8 จุด /  7 คน
table, chair, monitors                       6 จุด /  6 คน
```

**MiniLM-L12** — พังเร็วกว่า L6 ที่ mcs เดียวกัน เกิด mega-topic ทันที
```
courses, building, like, data, course       73 จุด / 54 คน  ← ยำหลักสูตรกับตึกรวมกัน
international, activities                   25 จุด / 19 คน
staff, nick, pdj                             20 จุด / 19 คน
good, clean                                   9 จุด /  9 คน
process, vm, request                          8 จุด /  8 คน
```
เหลือแค่ 5 topics เพราะ mega-topic กินไปเกือบครึ่งคอร์ปัส (73/145 = 50%)
นี่คือสาเหตุที่ npmi ของ L12 แย่กว่าที่ตารางบนดูเหมือนจะบอก — n_topics น้อยไม่ได้แปลว่าดี

**mpnet-base** — ใกล้เคียง L6 มากที่สุด แยก wifi ออกจาก facility ได้ดีกว่าด้วยซ้ำ
```
international, activities, thai              24 จุด / 18 คน
communication, nick, pdj, staff              18 จุด / 17 คน
courses, curriculum                          28 จุด / 16 คน  (เฟ้อ 1.75x)
monitors, table, old, plugs                  11 จุด / 10 คน
good, clean                                   8 จุด /  8 คน
wifi, cb2, internet                           7 จุด /  7 คน   ← แยกจาก facility ชัดกว่า L6
process, vm, slow                             7 จุด /  7 คน
printers, building                            6 จุด /  4 คน   (เฟ้อ 1.50x, reach ต่ำสุด)
```

## ข้อสรุป

**MiniLM-L12 ตกรอบ** — พังเร็วกว่า L6 ที่ mcs เดียวกัน (mega-topic เกิดที่ mcs=5 แทนที่จะเป็น mcs≥6)
ทั้งที่โมเดลใหญ่กว่าและช้ากว่า ไม่มีเหตุผลจะใช้แทน L6

**mpnet-base ให้ผลใกล้เคียง L6 มาก และแยก wifi ออกจาก facility ได้ดีกว่าเล็กน้อย**
แต่ใหญ่กว่า 4.5 เท่า (420MB vs 90MB) และมิติสูงกว่า (768 vs 384) ซึ่งมีผลกับขนาด
pgvector column ตอนเก็บลง database จริง — ต้องตัดสินใจตอน design schema

**บน corpus ขนาดนี้ (145 จุด) ความต่างระหว่าง L6 กับ mpnet ไม่มากพอจะคุ้มกับ
ต้นทุนที่เพิ่มขึ้น** L6 (ตัวปัจจุบัน) ยังเป็นตัวเลือกที่สมเหตุสมผลที่สุด
ควรกลับมาเทียบใหม่อีกครั้งเมื่อมีข้อมูลจริงจาก pilot (~370+ atomic points ตามที่
ประมาณไว้ใน docs/ai-pipeline-design.md) เพราะผลอาจเปลี่ยนที่ scale ใหญ่ขึ้น

## ยังไม่ได้ลอง

- `paraphrase-multilingual-MiniLM-L12-v2` — ตัวที่ ch3 อ้างถึง ยังไม่ได้เทียบ
- `BAAI/bge-base-en-v1.5` — คะแนน benchmark สูงกว่า mpnet ในหลาย task
- โมเดลตระกูล `e5` ต้องเติม prefix `"passage: "` ก่อน embed ซึ่ง `cluster_points.py`
  ยังไม่รองรับ ถ้าจะลองต้องแก้โค้ดเพิ่ม
