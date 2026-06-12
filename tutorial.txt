



## ขั้นที่ 1 — เข้าโฟลเดอร์โปรเจกต์


cd ~/blind_assist_ws


## ขั้นที่ 2 — ดึงโค้ดล่าสุดจาก GitHub


git pull origin main



## ขั้นที่ 3 — เริ่มเก็บภาพ (ใส่ชื่อตัวเอง)

เปลี่ยนชื่อ`bob` เป็นชื่อน้องแล้วรัน:


python3 record.py --name bob --count 50  --show 

ถ้ากล้องกลับหัว ให้รัน: 

python3 record.py --name bob --count 50  --show --no-rotate

โปรแกรมจะถ่ายอัตโนมัติทุก 3 วินาที จนครบ 50 ภาพ และพิมพ์บอกทุกครั้งที่ถ่าย:






