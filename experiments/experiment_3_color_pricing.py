# تجربة اكتشاف أكثر من لون للمنتجات وحساب السعر الإجمالي
import cv2
import numpy as np

# ===============================
# تعريف نطاقات الألوان بالـ HSV والسعر لكل لون
# ===============================
# نستخدم القواميس (Dictionaries) لتخزين خصائص كل لون (الحد الأدنى، الحد الأعلى، والسعر الوحدوي)
color_ranges = {
    'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255]), 'price': 2},  # اللون الأصفر بسعر 2
    'orange': {'lower': np.array([10, 100, 100]), 'upper': np.array([20, 255, 255]), 'price': 1},  # اللون البرتقالي بسعر 1
    'green':  {'lower': np.array([40, 50, 50]), 'upper': np.array([90, 255, 255]), 'price': 3},    # اللون الأخضر بسعر 3
}

# color_ranges = {

#     'yellow': {

#         'lower': np.array([15, 40, 40]),
#         'upper': np.array([40, 255, 255]),
#         'price': 2
#     },

#     'orange': {

#         'lower': np.array([5, 40, 40]),
#         'upper': np.array([22, 255, 255]),
#         'price': 1
#     },

#     'green': {

#         'lower': np.array([35, 30, 30]),
#         'upper': np.array([95, 255, 255]),
#         'price': 3
#     }

# }
# ===============================
# فتح كاميرا الجهاز
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("لم أستطع فتح الكاميرا")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تحويل الصورة إلى نظام HSV لزيادة دقة تمييز الألوان بعيداً عن تأثيرات الإضاءة
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    total_price = 0  # متغير لتجميع السعر الإجمالي لكل المنتجات في الكادر
    counts = {}      # قاموس لتخزين عدد الأجسام المكتشفة من كل لون

    # ===============================
    # حلقة تكرارية لفحص كل لون معرف في القائمة أعلاه
    # ===============================
    for color_name, props in color_ranges.items():
        # إنشاء قناع (Mask) يعزل اللون الحالي فقط
        mask = cv2.inRange(hsv, props['lower'], props['upper'])

        # تنظيف القناع من الضوضاء البيكسلية (النقاط الصغيرة التي ليست أجساماً حقيقية)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # البحث عن حدود الأجسام (Contours) الملونة في القناع المنظف
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0 # عداد خاص باللون الحالي
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # تجاهل أي كتلة لونية صغيرة جداً (أقل من 1000 بكسل) لتجنب الأخطاء
            if area > 1000:
                # رسم مستطيل حول الجسم المكتشف
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                count += 1 # زيادة عدد الأجسام لهذا اللون

        # تخزين النتيجة وحساب التكلفة لهذا اللون (العدد × السعر)
        counts[color_name] = count
        total_price += count * props['price']

    # ===============================
    # عرض النتائج النهائية على الشاشة
    # ===============================
    # تجميع معلومات الأعداد (مثلاً yellow:2 | green:1) لتحضيرها للكتابة
    info_text = " | ".join([f"{k}:{v}" for k,v in counts.items()])
    
    # كتابة النص الذي يحتوي على الأعداد والسعر الإجمالي في أعلى الصورة
    cv2.putText(frame, f"{info_text} | Total Price: {total_price}", 
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # إظهار النافذة للمستخدم
    cv2.imshow("Product Color Detection & Pricing", frame)

    # الخروج من البرنامج عند الضغط على مفتاح 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إغلاق الكاميرا والنوافذ عند الانتهاء
cap.release()
cv2.destroyAllWindows()