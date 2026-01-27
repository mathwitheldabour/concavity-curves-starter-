import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# 1. إعداد الصفحة
st.set_page_config(page_title="Concavity Master Class", layout="wide")

# تهيئة متغيرات الحالة (للتنقل وحساب الدرجات)
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'quiz_finished' not in st.session_state:
    st.session_state.quiz_finished = False

st.title("🎓 Training: Inflection Points & Concavity")
st.markdown("**Focus:** Determine the graph based on the sign of the second derivative $f''(x)$.")
st.markdown("---")

# مجال الرسم
x = np.linspace(-3, 3, 600)

# --- بنك الأسئلة (5 أسئلة مركزة على التقعر) ---
quiz_data = [
    # === السؤال 1: المفهوم الأساسي (تقعر لأعلى دائماً) ===
    {
        "title": "Basic Concavity",
        "desc_en": "Choose the graph that is Concave Up everywhere.",
        "math": [
            r"f''(x) > 0 \quad \forall x \in (-\infty, \infty)",
            r"f(0) = -2"
        ],
        "graphs": [
            lambda x: 0.5 * x**2 - 2,     # (A) صحيح: قطع مكافئ مفتوح لأعلى
            lambda x: -0.5 * x**2 - 2,    # (B) خطأ: مقعر لأسفل
            lambda x: x**3 - 2,           # (C) خطأ: يغير التقعر عند 0
            lambda x: np.abs(x) - 2       # (D) خطأ: ليس له تقعر (خطي)
        ],
        "correct_idx": 0,
        "feedback": "Correct! $f''(x) > 0$ means the graph is shaped like a cup (U-shape)."
    },
    
    # === السؤال 2: نقطة انقلاب عند الصفر ===
    {
        "title": "Inflection at Origin",
        "desc_en": "Identify the graph where concavity changes from Down to Up at x=0.",
        "math": [
            r"f''(x) < 0 \quad \text{for } x < 0",
            r"f''(x) > 0 \quad \text{for } x > 0",
            r"f(0) = 0"
        ],
        "graphs": [
            lambda x: -x**3,              # (A) خطأ: من أعلى لأسفل
            lambda x: x**3,               # (B) صحيح: دالة تكعيبية قياسية
            lambda x: x**2,               # (C) خطأ: لا يوجد انقلاب
            lambda x: np.sin(x*2)         # (D) مشتت: يشبه التكعيبية لكنه دوري
        ],
        "correct_idx": 1,
        "feedback": "Correct! $f(x)=x^3$ changes from concave down to concave up at x=0."
    },

    # === السؤال 3: انقلاب عند نقطة غير الصفر (إزاحة) ===
    {
        "title": "Shifted Inflection Point",
        "desc_en": "Find the graph with an inflection point at **x = 1**.",
        "math": [
            r"f''(x) > 0 \quad \text{for } x < 1",
            r"f''(x) < 0 \quad \text{for } x > 1",
            r"f(1) = 0"
        ],
        "graphs": [
            lambda x: (x-1)**3,           # (A) خطأ: التقعر بالعكس (سالب ثم موجب)
            lambda x: -(x-1)**3,          # (B) صحيح: تقعر موجب ثم سالب حول 1
            lambda x: -(x+1)**3,          # (C) خطأ: الانقلاب عند -1
            lambda x: -(x-1)**2           # (D) خطأ: دالة زوجية حول 1
        ],
        "correct_idx": 1,
        "feedback": "Correct! The negative cubic function shifted to x=1."
    },

    # === السؤال 4: الجرس (قعر ثم قمة ثم قعر) ===
    {
        "title": "Complex Concavity (Bell Shape)",
        "desc_en": "Select the graph that is Concave Down in the middle, and Concave Up at the ends.",
        "math": [
            r"f''(x) < 0 \quad \text{for } x \in (-1, 1)",
            r"f''(x) > 0 \quad \text{for } |x| > 1",
            r"f'(0) = 0 \quad (\text{Max})"
        ],
        "graphs": [
            lambda x: x**4 - 2*x**2,      # (A) خطأ: مقعر لأعلى في الوسط (W shape)
            lambda x: 2 * np.exp(-x**2),  # (B) صحيح: دالة جرسية
            lambda x: -x**2,              # (C) خطأ: مقعر لأسفل دائماً
            lambda x: 1/(x**2+0.5)        # (D) مشابه للصحيح لكن أضيق (مقبول كخيار مشتت)
        ],
        # للتوضيح: الدالة 1/(x^2+1) تشبه الجرس أيضاً، لذا سنغير D لشيء مختلف
        # سنجعل D دالة تتذبذب
        "graphs": [
            lambda x: x**2 - 1,           # (A)
            lambda x: 3 * np.exp(-0.5*x**2), # (B) صحيح
            lambda x: -0.5*x**4 + x**2,   # (C) يشبه الجرس لكن مسطح
            lambda x: np.cos(x)           # (D)
        ],
        "correct_idx": 1,
        "feedback": "Correct! The Gaussian function (Bell Curve) is concave down near the peak."
    },

    # === السؤال 5: تمرين 37 (الأصعب) ===
    {
        "title": "Exercise 37 Challenge",
        "desc_en": "The ultimate test: Match the complex concavity conditions.",
        "math": [
            r"f''(x) > 0 \quad x < -1",
            r"f''(x) < 0 \quad -1 < x < 0",
            r"f''(x) > 0 \quad 0 < x < 1",
            r"f''(x) > 0 \quad x > 1 \quad (\text{Yes, still up})"
        ],
        "graphs": [
            # سنقوم ببناء دوال "Piecewise" ناعمة هنا للخيارات
            # (A) خطأ: مقعر لأعلى دائماً بين -1 و 1
            lambda x: np.piecewise(x, [x<0, x>=0], [lambda z: z**2, lambda z: z**2]), 
            
            # (B) خطأ: يقلب التقعر عند 1 (يصبح محدب)
            lambda x: x**3 - 3*x, 

            # (C) صحيح: يحقق الشروط (بنيناه سابقاً)
            lambda x: 3*x * np.exp(-0.5 * x**2) + 0.5 * np.arctan(x+1),
            
            # (D) خطأ: قمة مبكرة
            lambda x: 3*(x+0.5) * np.exp(-0.5 * (x+0.5)**2)
        ],
        "correct_idx": 2,
        "feedback": "Correct! Inflection points at x=-1 and x=0, and maintains Concave Up behavior for x>1."
    }
]

# --- منطق العرض ---

# عرض شريط التقدم
progress = (st.session_state.current_q) / len(quiz_data)
st.progress(progress)

if st.session_state.quiz_finished:
    # شاشة النهاية
    st.markdown(f"""
    <div style="text-align: center; padding: 40px; background-color: #f0f8ff; border-radius: 15px; border: 2px solid #0066cc;">
        <h1 style="color: #004080;">🏆 النتيجة النهائية</h1>
        <h2 style="font-size: 50px;">{st.session_state.score} / 5</h2>
        <p style="font-size: 20px;">لقد أكملت تدريب التقعر ونقاط الانقلاب.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 إعادة التدريب"):
        st.session_state.current_q = 0
        st.session_state.score = 0
        st.session_state.quiz_finished = False
        st.rerun()

else:
    # جلب السؤال الحالي
    q = quiz_data[st.session_state.current_q]
    
    st.subheader(f"Question {st.session_state.current_q + 1}: {q['title']}")
    
    # تقسيم الشاشة: يسار إنجليزي - يمين معادلات
    col_en, col_math = st.columns(2)
    
    with col_en:
        st.markdown(f"**{q['desc_en']}**")
    
    with col_math:
        # حاوية للمعادلات بتنسيق تمرين 37
        st.markdown('<div style="direction: rtl; text-align: right; border-right: 4px solid #ddd; padding-right: 10px;">', unsafe_allow_html=True)
        st.markdown("**المعطيات:**")
        for eq in q['math']:
            st.latex(eq)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # رسم الخيارات (4 رسومات)
    cols = st.columns(4)
    labels = ["A", "B", "C", "D"]
    
    for i in range(4):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            
            # رسم الدالة
            try:
                y_vals = q['graphs'][i](x)
            except:
                # Fallback for complex lambda functions if simple vectorization fails
                y_vals = np.array([q['graphs'][i](val) for val in x])

            # تنظيف القيم الزائدة للرسم (Clipping)
            y_vals = np.clip(y_vals, -4, 4)
            
            ax.plot(x, y_vals, color='#005580', linewidth=2.5)
            
            # محاور في المنتصف
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            
            # شبكة
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            
            # إخفاء الأرقام ما عدا المهمة
            ax.set_xticks([-1, 1])
            ax.set_yticks([])
            
            ax.set_title(labels[i], fontsize=14, fontweight='bold')
            st.pyplot(fig)

    # منطقة التفاعل
    st.markdown("### Select Graph:")
    choice = st.radio("Answer:", labels, horizontal=True, label_visibility="collapsed", key=f"q_{st.session_state.current_q}")
    
    if st.button("✅ Check & Next"):
        choice_idx = labels.index(choice)
        if choice_idx == q['correct_idx']:
            st.success(f"🎉 {q['feedback']}")
            st.session_state.score += 1
        else:
            st.error(f"❌ Incorrect. The correct answer was {labels[q['correct_idx']]}.")
        
        # تأخير بسيط للانتقال
        st.session_state.current_q += 1
        if st.session_state.current_q >= len(quiz_data):
            st.session_state.quiz_finished = True
        
        st.button("Next ➡️") # زر لتحديث الصفحة والانتقال