#!/usr/bin/env bash
#
# bt-connect.sh — เครื่องมือจัดการบลูทูธผ่านเทอร์มินัล (สำหรับ Raspberry Pi / SSH)
#
# เปิดอะแดปเตอร์บลูทูธ, สแกนหาอุปกรณ์รอบ ๆ, เลือกจากรายการ แล้วจับคู่/เชื่อมต่อ
# หรือเลือกตัดการเชื่อมต่อก็ได้ ออกแบบมาให้ใช้แบบไม่มีจอผ่าน SSH บน Raspberry Pi
#
# วิธีใช้:
#   ./bt-connect.sh                       # สแกน + เลือก + เชื่อมต่อ
#   ./bt-connect.sh -s 15                  # สแกน 15 วินาที (ค่าเริ่มต้น 10)
#   ./bt-connect.sh -m AA:BB:CC:DD:EE:FF   # เชื่อมต่อตรงไปยัง MAC ไม่ต้องเลือกเมนู
#   ./bt-connect.sh disconnect            # เมนูอุปกรณ์ที่เชื่อมอยู่ เลือกเพื่อตัดการเชื่อมต่อ
#   ./bt-connect.sh disconnect -m MAC     # ตัดการเชื่อมต่อตรงไปยัง MAC ไม่ต้องเลือกเมนู
#
set -uo pipefail

SCAN_SECS=10
TARGET_MAC=""
DISCONNECT=0

usage() {
    cat <<EOF
วิธีใช้: $0 [disconnect] [-s วินาที] [-m MAC]
  disconnect   โหมดตัดการเชื่อมต่อ (เลือกจากอุปกรณ์ที่เชื่อมอยู่ หรือใช้ร่วมกับ -m)
  -s วินาที    ระยะเวลาสแกนหาอุปกรณ์ (ค่าเริ่มต้น: $SCAN_SECS)
  -m MAC       ระบุ MAC โดยตรง ข้ามการเลือกเมนู
  -h           แสดงวิธีใช้
EOF
    exit 0
}

# รองรับคำสั่งย่อย "disconnect" เป็นอาร์กิวเมนต์แรก
if [[ "${1:-}" == "disconnect" ]]; then
    DISCONNECT=1
    shift
fi

while getopts "s:m:h" opt; do
    case "$opt" in
        s) SCAN_SECS="$OPTARG" ;;
        m) TARGET_MAC="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# --- ตรวจสอบความพร้อม -------------------------------------------------------
if ! command -v bluetoothctl >/dev/null 2>&1; then
    echo "ข้อผิดพลาด: ไม่พบ bluetoothctl ติดตั้งด้วยคำสั่ง:" >&2
    echo "  sudo apt-get install -y bluez" >&2
    exit 1
fi

# ตรวจสอบว่าบริการบลูทูธทำงานอยู่หรือไม่
if command -v systemctl >/dev/null 2>&1; then
    if ! systemctl is-active --quiet bluetooth; then
        echo ">> กำลังเริ่มบริการบลูทูธ..."
        sudo systemctl start bluetooth || true
        sleep 1
    fi
fi

# --- เปิดอะแดปเตอร์ ----------------------------------------------------------
echo ">> กำลังเปิดอะแดปเตอร์บลูทูธ..."
bluetoothctl power on >/dev/null
bluetoothctl agent on >/dev/null
bluetoothctl default-agent >/dev/null 2>&1

connect_device() {
    local mac="$1"
    echo
    echo ">> กำลังจับคู่กับ $mac ..."
    bluetoothctl pair "$mac"        2>/dev/null || true
    echo ">> กำลังตั้งค่าเชื่อถือ $mac (เชื่อมต่ออัตโนมัติเมื่อบูต) ..."
    bluetoothctl trust "$mac"       >/dev/null 2>&1 || true
    echo ">> กำลังเชื่อมต่อกับ $mac ..."
    if bluetoothctl connect "$mac"; then
        echo
        echo "✓ เชื่อมต่อกับ $mac สำเร็จ"
    else
        echo
        echo "✗ เชื่อมต่อกับ $mac ไม่สำเร็จ" >&2
        return 1
    fi
}

disconnect_device() {
    local mac="$1"
    echo
    echo ">> กำลังตัดการเชื่อมต่อจาก $mac ..."
    if bluetoothctl disconnect "$mac"; then
        echo
        echo "✓ ตัดการเชื่อมต่อจาก $mac สำเร็จ"
    else
        echo
        echo "✗ ตัดการเชื่อมต่อจาก $mac ไม่สำเร็จ" >&2
        return 1
    fi
}

# --- โหมดระบุ MAC ตรง (ไม่ต้องเลือกเมนู) ------------------------------------
if [[ -n "$TARGET_MAC" ]]; then
    if [[ "$DISCONNECT" -eq 1 ]]; then
        disconnect_device "$TARGET_MAC"
    else
        connect_device "$TARGET_MAC"
    fi
    exit $?
fi

# --- โหมดเมนูตัดการเชื่อมต่อ -------------------------------------------------
if [[ "$DISCONNECT" -eq 1 ]]; then
    # แสดงเฉพาะอุปกรณ์ที่กำลังเชื่อมต่ออยู่
    mapfile -t CONNECTED < <(bluetoothctl devices Connected | sed -n 's/^Device //p')

    if [[ ${#CONNECTED[@]} -eq 0 ]]; then
        echo "ไม่มีอุปกรณ์ที่เชื่อมต่ออยู่ในขณะนี้"
        exit 0
    fi

    echo
    echo "อุปกรณ์ที่เชื่อมต่ออยู่:"
    echo "--------------------------------------------"
    i=1
    declare -a CMACS=()
    for entry in "${CONNECTED[@]}"; do
        mac="${entry%% *}"
        name="${entry#* }"
        [[ "$name" == "$mac" ]] && name="(ไม่ทราบชื่อ)"
        printf "  %2d) %-17s  %s\n" "$i" "$mac" "$name"
        CMACS+=("$mac")
        ((i++))
    done
    echo "--------------------------------------------"

    read -rp "เลือกหมายเลขอุปกรณ์ที่จะตัดการเชื่อมต่อ (หรือ q เพื่อออก): " choice
    if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
        echo "ยกเลิกแล้ว"
        exit 0
    fi
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#CMACS[@]} )); then
        echo "การเลือกไม่ถูกต้อง" >&2
        exit 1
    fi
    disconnect_device "${CMACS[$((choice-1))]}"
    exit $?
fi

# --- สแกนหาอุปกรณ์ ----------------------------------------------------------
echo ">> กำลังสแกน $SCAN_SECS วินาที... (เปิดให้อุปกรณ์อยู่ในโหมดค้นหาได้)"
bluetoothctl --timeout "$SCAN_SECS" scan on >/dev/null 2>&1

# เก็บรายการอุปกรณ์ที่พบ/รู้จัก รูปแบบจาก bluetoothctl: "Device <MAC> <NAME>"
mapfile -t DEVICES < <(bluetoothctl devices | sed -n 's/^Device //p')

if [[ ${#DEVICES[@]} -eq 0 ]]; then
    echo "ไม่พบอุปกรณ์ ตรวจสอบว่าอุปกรณ์เปิดอยู่และอยู่ในโหมดค้นหา แล้วลองใหม่อีกครั้ง" >&2
    exit 1
fi

# --- แสดงเมนูแบบหมายเลข ------------------------------------------------------
echo
echo "อุปกรณ์ที่พบ:"
echo "--------------------------------------------"
i=1
declare -a MACS=()
for entry in "${DEVICES[@]}"; do
    mac="${entry%% *}"           # ฟิลด์แรก
    name="${entry#* }"           # ส่วนที่เหลือของบรรทัด
    [[ "$name" == "$mac" ]] && name="(ไม่ทราบชื่อ)"
    printf "  %2d) %-17s  %s\n" "$i" "$mac" "$name"
    MACS+=("$mac")
    ((i++))
done
echo "--------------------------------------------"

# --- ให้ผู้ใช้เลือก ---------------------------------------------------------
read -rp "เลือกหมายเลขอุปกรณ์ที่จะเชื่อมต่อ (หรือ q เพื่อออก): " choice
if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
    echo "ยกเลิกแล้ว"
    exit 0
fi

if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#MACS[@]} )); then
    echo "การเลือกไม่ถูกต้อง" >&2
    exit 1
fi

SELECTED_MAC="${MACS[$((choice-1))]}"
connect_device "$SELECTED_MAC"
