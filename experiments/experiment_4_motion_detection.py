import cv2
import numpy as np

# تحميل ملف الفيديو من المسار المحدد
cap = cv2.VideoCapture("C:\\Users\\Zaytona\\Videos\\Captures\\v3.WMV")

# إعداد محرك "طرح الخلفية" (Background Subtraction)
# هذا المحرك يقوم ببناء نموذج للمشهد الثابت ويعتبر أي تغيير مفاجئ "حركة"
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,       # عدد الفريمات السابقة التي يعتمد عليها المحرك لفهم الخلفية
    varThreshold=120,  # عتبة الحساسية؛ رفعنا القيمة هنا ليتجاهل الاهتزازات البسيطة للكاميرا
    detectShadows=False # إلغاء كشف الظلال لتجنب اعتبار ظل السيارة كجسم متحرك
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تغيير أبعاد الفيديو لتسريع عملية المعالجة وتوحيد العرض
    frame = cv2.resize(frame, (800, 450))
    
    # تحويل الصورة لرمادي ثم تطبيق "تنعيم غاوسي" (Gaussian Blur)
    # هذه الخطوة أساسية لتقليل "الضجيج الرقمي" واهتزاز البكسلات الناتج عن حركة الكاميرا
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)

    # تطبيق المحرك على الصورة المنعمة للحصول على "قناع الحركة" (Foreground Mask)
    fg_mask = bg_subtractor.apply(gray)
    
    # تحويل القناع لصورة ثنائية (أبيض وأسود فقط) لتحديد الأجسام المتحركة بوضوح
    _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

    # تحسين شكل القناع باستخدام العمليات المورفولوجية
    # MORPH_CLOSE: لدمج الأجزاء المتقطعة من الجسم الواحد (مثل ربط سقف السيارة بأسفلها)
    # dilate: تضخيم الكتلة البيضاء لضمان إحاطة المستطيل بالجسم كاملاً
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    # البحث عن الحدود الخارجية (Contours) للكتل البيضاء المكتشفة في القناع
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # حساب مساحة الكتلة؛ إذا كانت أصغر من 3000 بكسل نعتبرها ضوضاء (مثل اهتزاز شجر أو رصيف)
        if cv2.contourArea(cnt) < 3000: 
            continue

        # تحديد أبعاد المربع الذي يحيط بالكتلة المتحركة
        x, y, w, h = cv2.boundingRect(cnt)

        # رسم المستطيل الأخضر حول الجسم المتحرك في الصورة الأصلية
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # عرض النافذة النهائية التي تظهر المربعات حول الأجسام المتحركة فقط
    cv2.imshow("Motion Detection Only", frame)

    # الخروج من البرنامج عند الضغط على مفتاح 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()