# كود عد السيارات الإجمالي وعد السيارات ذات اللون الأحمر
import cv2
import numpy as np

# تحميل الفيديو من المسار المحدد
cap = cv2.VideoCapture("C:\\Users\\Zaytona\\Videos\\Captures\\v1.WMV")

# إنشاء محرك طرح الخلفية (Background Subtraction) لعزل الأجسام المتحركة عن الطريق
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,        # عدد الفريمات التي يتذكرها المحرك لبناء نموذج الخلفية
    varThreshold=80,    # عتبة الحساسية (كلما زادت قل الضجيج المكتشف)
    detectShadows=False # إيقاف كشف الظلال لزيادة سرعة الأداء
)

# تعريف العدادات وإحداثيات خط العد
car_count = 0
red_car_count = 0
line_position = 225  # موقع الخط الوهمي الذي عند تجاوزه يتم عد السيارة

detected_ids = set() # مجموعة لتخزين المعرفات الفريدة للأجسام لمنع تكرار عدها
object_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تغيير حجم الصورة لضمان سرعة المعالجة وتوحيد الأبعاد
    frame = cv2.resize(frame, (800,450))

    # تطبيق محرك طرح الخلفية للحصول على قناع يظهر الأجسام المتحركة فقط
    fg_mask = bg_subtractor.apply(frame)

    # تحويل القناع إلى صورة ثنائية (أبيض وأسود فقط) لإزالة التدرجات الرمادية والضجيج
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # تحسين القناع باستخدام العمليات المورفولوجية (فتح وتمدد) لربط أجزاء السيارة ببعضها
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel) # إزالة النمش الصغير
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)        # تضخيم الأجسام لسد الفجوات

    # إيجاد الخطوط الخارجية (Contours) لكل جسم متحرك في القناع
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # تجاهل الأجسام الصغيرة جداً التي لا تمثل سيارات (مثل المشاة أو ضجيج الكاميرا)
        if cv2.contourArea(cnt) < 2500:
            continue

        # الحصول على إحداثيات المربع المحيط بالسيارة
        x, y, w, h = cv2.boundingRect(cnt)
        
        # حساب نقطة المنتصف الرأسية للسيارة (Center Y)
        center_y = y + h // 2

        # التحقق إذا كانت السيارة تمر الآن فوق "خط العد" الوهمي
        if abs(center_y - line_position) < 5:
            # التأكد من أن هذه السيارة لم يتم رصدها وعدّها من قبل
            if object_id not in detected_ids:
                detected_ids.add(object_id)
                car_count += 1 # زيادة العداد العام للسيارات

                # --- منطق كشف اللون الأحمر ---
                # استقطاع صورة السيارة فقط (ROI) لتحليل لونها
                roi = frame[y:y+h, x:x+w]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # تعريف نطاقي اللون الأحمر (لأن الأحمر يظهر في بداية ونهاية تدريج HSV)
                lower_red1, upper_red1 = np.array([0,120,70]), np.array([10,255,255])
                lower_red2, upper_red2 = np.array([170,120,70]), np.array([180,255,255])

                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = mask1 + mask2 # دمج القناعين

                # إذا كان مجموع البكسلات الحمراء كبيراً كفاية، نعتبرها سيارة حمراء
                if np.sum(red_mask) > 5000:
                    red_car_count += 1

                object_id += 1 # الانتقال للمعرف التالي

        # رسم مستطيل أخضر حول كل سيارة متحركة تظهر في الكادر
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # طباعة النتائج (العدادات) على الفريم مباشرة
    cv2.putText(frame, "Cars: " + str(car_count), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Red Cars: " + str(red_car_count), (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # عرض الفيديو النهائي مع المربعات والعدادات
    cv2.imshow("Vehicle Counter", frame)

    # الخروج عند الضغط على مفتاح 'q' أو انتظار 30 ميلي ثانية بين الفريمات
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# إغلاق مصادر الفيديو والنوافذ
cap.release()
cv2.destroyAllWindows()