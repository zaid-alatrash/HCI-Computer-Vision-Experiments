# كود اكتشاف المنتج إذا كان صالحاً أو فاسداً بناءً على تحليل اللون
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
last_rects = {}
alpha = 0.3

# نطاق التتبع (Track Range): يدمج الأصفر والبني معاً لضمان أن يظل المستطيل يحيط بالثمرة حتى لو تغير لونها بالكامل
track_lower = np.array([0, 15, 15])
track_upper = np.array([50, 255, 255])

# نطاق اللون البني فقط: يستخدم داخل المربع المكتشف لفحص وجود بقع عفن أو تلف
brown_lower = np.array([3, 80, 20])
brown_upper = np.array([16, 255, 120])

while True:
    ret, frame = cap.read()
    if not ret: break

    # تحويل الصورة إلى نظام HSV لتسهيل عزل درجات الألوان بدقة
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # إنشاء قناع (Mask) لملاحقة الجسم (يشمل درجات الأصفر والبرتقالي والبني)
    track_mask = cv2.inRange(hsv, track_lower, track_upper)

    # تنظيف القناع من الشوائب الناتجة عن الإضاءة باستخدام العمليات المورفولوجية
    kernel = np.ones((5,5), np.uint8)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel) # حذف النقاط الصغيرة
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel) # ملء الثقوب داخل الجسم

    # استخراج الخطوط الخارجية (Contours) للأجسام المكتشفة
    contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # فلترة الأجسام بناءً على المساحة (تجاهل أي جسم أصغر من 2000 بكسل)
        if area > 2000:
            current_rects.append(cv2.boundingRect(cnt))

    new_last_rects = {}
    for i, rect in enumerate(current_rects):
        # تطبيق التنعيم (Alpha Smoothing) لضمان ثبات المستطيل ومنع اهتزازه أثناء الحركة
        if i in last_rects:
            prev = last_rects[i]
            smoothed = [
                int(prev[0]*(1-alpha)+rect[0]*alpha),
                int(prev[1]*(1-alpha)+rect[1]*alpha),
                int(prev[2]*(1-alpha)+rect[2]*alpha),
                int(prev[3]*(1-alpha)+rect[3]*alpha)
            ]
            new_last_rects[i] = smoothed
        else:
            new_last_rects[i] = list(rect)

        rx, ry, rw, rh = new_last_rects[i]
        
        # استقطاع منطقة الجسم فقط (ROI) لتحليلها لونياً بشكل مستقل عن باقي الصورة
        roi_hsv = hsv[ry:ry+rh, rx:rx+rw]
        
        # إنشاء قناع خاص باللون البني داخل منطقة الجسم فقط
        brown_pixel_mask = cv2.inRange(roi_hsv, brown_lower, brown_upper)
        
        # حساب عدد البكسلات البنية المكتشفة
        brown_count = cv2.countNonZero(brown_pixel_mask)
        # حساب المساحة الكلية للمربع المكتشف
        total_pixels = rw * rh
        # حساب النسبة المئوية للون البني بالنسبة لحجم الجسم
        brown_ratio = (brown_count / total_pixels) * 100

        # اتخاذ القرار: إذا كانت نسبة اللون البني > 1%، يُصنف كمنتج فاسد (Bad)
        if brown_ratio > 1:  
            status = f"Bad"
            color = (0, 0, 255) # لون أحمر للتنبيه
        else:
            status = "Good"
            color = (0, 255, 0) # لون أخضر للمنتج السليم

        # رسم المستطيل وكتابة الحالة (صالحة/فاسدة) فوق الجسم مباشرة
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, 3)
        cv2.putText(frame, status, (rx, ry-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # تحديث الذاكرة لاستخدامها في تنعيم الفريم القادم
    last_rects = new_last_rects
    cv2.imshow("Lemon Quality Check", frame)

    # التوقف عند الضغط على مفتاح 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()