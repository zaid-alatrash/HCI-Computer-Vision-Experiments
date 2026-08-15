# تجربة حساب كم جسم لونه أصفر 
import cv2
import numpy as np
from util import get_limits

# تحديد اللون المستهدف بتنسيق BGR (أصفر)
target_color_bgr = [0, 255, 255] 
cap = cv2.VideoCapture(0)

# قاموس لتخزين إحداثيات المستطيلات السابقة لغرض التنعيم، ومعامل التنعيم alpha
last_rects = {} 
alpha = 0.3

while True:
    ret, frame = cap.read()
    if not ret: break

    # تحويل الصورة من BGR إلى HSV لأن التعامل مع الألوان فيها أدق
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # استخراج حدود اللون (أقل وأعلى قيمة) بناءً على اللون المطلوب
    lowerLimit, upperLimit = get_limits(color=target_color_bgr)
    
    # إنشاء "قناع" يظهر الأماكن التي يظهر فيها اللون المطلوب فقط باللون الأبيض والباقي أسود
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # عمليات مورفولوجية (تنظيف القناع) لإزالة النقاط البيضاء الصغيرة وسد الفجوات داخل الأجسام
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # إزالة الضجيج
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # سد الفتحات

    # البحث عن "الكنتور" أو الحدود الخارجية لكل كتلة لونية منفصلة في القناع
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yellow_count = 0 # عداد الأجسام الصفراء
    current_rects = [] # قائمة لتخزين المستطيلات المكتشفة في الفريم الحالي

    for cnt in contours:
        # حساب مساحة الكتلة اللونية لتجنب عد النقاط الصغيرة جداً كأجسام
        area = cv2.contourArea(cnt)
        if area > 1000:
            # الحصول على إحداثيات المستطيل المحيط بالكتلة (السين، الصاد، العرض، الطول)
            x, y, w, h = cv2.boundingRect(cnt)
            current_rects.append((x, y, w, h))
            yellow_count += 1 # زيادة العداد عند اكتشاف جسم يحقق الشروط

    new_last_rects = {} # لتخزين المستطيلات الجديدة بعد "التنعيم"
    for i, rect in enumerate(current_rects):
        # إذا كان الجسم موجوداً في الفريم السابق، يتم دمج الإحداثيات لتقليل الاهتزاز (Smoothing)
        if i in last_rects:
            prev_rect = last_rects[i]
            smoothed = [
                int(prev_rect[0] * (1 - alpha) + rect[0] * alpha),
                int(prev_rect[1] * (1 - alpha) + rect[1] * alpha),
                int(prev_rect[2] * (1 - alpha) + rect[2] * alpha),
                int(prev_rect[3] * (1 - alpha) + rect[3] * alpha)
            ]
            new_last_rects[i] = smoothed
        else:
            new_last_rects[i] = list(rect)

        # رسم المستطيل الأخضر حول كل جسم بناءً على الإحداثيات المنعمة
        rx, ry, rw, rh = new_last_rects[i]
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 4)

    # تحديث قائمة المستطيلات للفريم القادم
    last_rects = new_last_rects

    # كتابة عدد الأجسام الصفراء المكتشفة على الشاشة
    cv2.putText(frame, f"Yellow objects: {yellow_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # عرض النتيجة النهائية
    cv2.imshow('Multi-Object Stable Detection', frame)

    # الخروج من البرنامج عند الضغط على حرف 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()