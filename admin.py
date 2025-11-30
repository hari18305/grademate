from flask import Flask, render_template, request, redirect, url_for, session, flash
import psycopg2
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from collections import defaultdict
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import language_tool_python

warnings.filterwarnings("ignore")

# NLTK downloads (run once; if already downloaded they are skipped)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("vader_lexicon")

device = torch.device("cpu")

# Load model only once (globally)
model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=str(device))

app = Flask(__name__)
app.secret_key = "mysecretkey123"

# Set the template folder
app.template_folder = "templates"

# PostgreSQL configuration
def get_db():
    return psycopg2.connect(
        host="localhost",
        database="teacher_part",
        user="postgres",            # change if needed
        password="1234" # <<< CHANGE THIS
    )

# Set English stopwords
EN_STOPWORDS = set(stopwords.words("english"))

# ---------------------- TEXT PROCESSING & SCORING HELPERS ---------------------- #

def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]  # Lemmatization
    return lemmatized_tokens

# Exact Match function
def exact_match(expected_answer, student_answer):
    return int(expected_answer == student_answer)

# Partial Match function
def partial_match(expected_answer, student_answer):
    expected_tokens = preprocess_text(expected_answer)
    student_tokens = preprocess_text(student_answer)
    common_tokens = set(expected_tokens) & set(student_tokens)
    match_percentage = len(common_tokens) / max(len(expected_tokens), len(student_tokens))
    return match_percentage

# Cosine Similarity function
def cosine_similarity_score(expected_answer, student_answer):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform([expected_answer, student_answer])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim

# Sentiment Analysis function
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)["compound"]
    return (sentiment_score + 1) / 2  # Normalize to range [0, 1]

# Function to calculate enhanced sentence match score using Semantic Similarity
def enhanced_sentence_match(expected_answer, student_answer):
    embeddings_expected = model.encode([expected_answer])
    embeddings_student = model.encode([student_answer])
    similarity = cosine_similarity(
        [embeddings_expected.flatten()],
        [embeddings_student.flatten()]
    )[0][0]
    return similarity

# Function to calculate multinomial naive Bayes score
def multinomial_naive_bayes_score(expected_answer, student_answer):
    answers = [expected_answer, student_answer]
    vectorizer = CountVectorizer(tokenizer=preprocess_text)
    X = vectorizer.fit_transform(answers)
    y = [0, 1]  # 0 for expected_answer, 1 for student_answer
    clf = MultinomialNB()
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    return probs[1][1]

# Function to calculate weighted average score
def weighted_average_score(scores, weights):
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

def semantic_similarity_score(expected_answer, student_answer):
    embeddings_expected = model.encode([expected_answer])
    embeddings_student = model.encode([student_answer])
    similarity = cosine_similarity(
        [embeddings_expected.flatten()],
        [embeddings_student.flatten()]
    )[0][0]
    return similarity

def coherence_score(expected_answer, student_answer):
    len_expected = len(word_tokenize(expected_answer))
    len_student = len(word_tokenize(student_answer))
    coherence_score_val = min(len_expected, len_student) / max(len_expected, len_student)
    return coherence_score_val

def relevance_score(expected_answer, student_answer):
    expected_tokens = set(word_tokenize(expected_answer.lower()))
    student_tokens = set(word_tokenize(student_answer.lower()))
    common_tokens = expected_tokens.intersection(student_tokens)
    relevance_score_val = len(common_tokens) / len(expected_tokens) if expected_tokens else 0
    return relevance_score_val

def evaluate(expected, response):
    if expected == response:
        return 10
    elif not response:
        return 0

    exact_match_score_val = exact_match(expected, response)
    partial_match_score_val = partial_match(expected, response)
    cosine_similarity_score_value = cosine_similarity_score(expected, response)
    sentiment_score = sentiment_analysis(response)
    enhanced_sentence_match_score = enhanced_sentence_match(expected, response)
    multinomial_naive_bayes_score_value = multinomial_naive_bayes_score(expected, response)
    semantic_similarity_value = semantic_similarity_score(expected, response)
    coherence_value = coherence_score(expected, response)
    relevance_value = relevance_score(expected, response)

    scores = [
        exact_match_score_val,
        partial_match_score_val,
        cosine_similarity_score_value,
        sentiment_score,
        enhanced_sentence_match_score,
        multinomial_naive_bayes_score_value,
        semantic_similarity_value,
        coherence_value,
        relevance_value,
    ]
    weights = [0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]

    scaled_scores = [score * 10 for score in scores]
    final_score = weighted_average_score(scaled_scores, weights)
    rounded_score = round(final_score)

    marks = rounded_score
    return marks

# ---------------------- BASIC ROUTES ---------------------- #

@app.route("/")
def index():
    return render_template("Homepage.html")

# ---------------------- ADMIN ROUTES ---------------------- #

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM admins WHERE username = %s AND password = %s",
            (username, password),
        )
        admin = cur.fetchone()
        cur.close()
        conn.close()

        if admin:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_home"))
        else:
            return render_template("adminlogin.html", error="Invalid username or password")

    return render_template("adminlogin.html")

@app.route("/admin/home")
def admin_home():
    if "admin_logged_in" in session:
        return render_template("adminhome.html")
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/students")
def admin_students():
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM students")
        students = cur.fetchall()
        cur.close()
        conn.close()
        return render_template("admin_students.html", students=students)
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/add_student", methods=["POST"])
def add_student():
    if "admin_logged_in" in session:
        username = request.form["username"]
        password = request.form["password"]
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO students (username, password) VALUES (%s, %s)",
            (username, password),
        )
        conn.commit()
        cur.close()
        conn.close()
        return redirect(url_for("admin_students"))
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/update_student/<int:student_id>", methods=["POST"])
def update_student(student_id):
    if "admin_logged_in" in session:
        username = request.form["username"]
        password = request.form["password"]
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "UPDATE students SET username = %s, password = %s WHERE student_id = %s",
            (username, password, student_id),
        )
        conn.commit()
        cur.close()
        conn.close()
        return redirect(url_for("admin_students"))
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/delete_student/<int:student_id>", methods=["POST"])
def delete_student(student_id):
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM students WHERE student_id = %s", (student_id,))
        conn.commit()
        cur.close()
        conn.close()
        return redirect(url_for("admin_students"))
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/view_student_scores/<int:student_id>")
def view_student_scores(student_id):
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        query = """
            SELECT DISTINCT sa.answer_id, sa.test_id, t.test_name, q.question_text, 
                ea.answer_text AS expected_answer, 
                sa.answer_text AS student_answer, sa.score
            FROM studentanswers sa
            JOIN tests t ON sa.test_id = t.test_id
            JOIN questions q ON sa.question_id = q.question_id
            JOIN expectedanswers ea ON q.question_id = ea.question_id
            WHERE sa.student_id = %s
            ORDER BY sa.test_id, q.question_id;
        """
        cur.execute(query, (student_id,))
        scores = cur.fetchall()
        cur.close()
        conn.close()

        scores = [
            {
                "answer_id": score[0],
                "test_id": score[1],
                "test_name": score[2],
                "question_text": score[3],
                "expected_answer": score[4],
                "student_answer": score[5],
                "score": score[6],
            }
            for score in scores
        ]
        return render_template("student_scores.html", scores=scores)
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/delete_student_score/<int:answer_id>", methods=["POST"])
def delete_student_score(answer_id):
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        query = "DELETE FROM studentanswers WHERE answer_id = %s"
        cur.execute(query, (answer_id,))
        conn.commit()
        cur.close()
        conn.close()
        return redirect(url_for("admin_students"))
    else:
        return redirect(url_for("admin_login"))

# ---------------------- ADMIN - TEACHERS ---------------------- #

@app.route("/admin/teachers")
def admin_teachers():
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM teachers")
        teachers = cur.fetchall()
        cur.close()
        conn.close()
        return render_template("admin_teachers.html", teachers=teachers)
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/add_teacher", methods=["GET", "POST"])
def add_teacher():
    if "admin_logged_in" in session:
        if request.method == "POST":
            username = request.form["username"]
            password = request.form["password"]
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO teachers (username, password) VALUES (%s, %s)",
                (username, password),
            )
            conn.commit()
            cur.close()
            conn.close()
            return redirect(url_for("admin_teachers"))
        else:
            return render_template("add_teacher.html")
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/update_teacher/<int:teacher_id>", methods=["GET", "POST"])
def update_teacher(teacher_id):
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        if request.method == "POST":
            try:
                username = request.form["username"]
                password = request.form["password"]
                cur.execute(
                    "UPDATE teachers SET username = %s, password = %s WHERE teacher_id = %s",
                    (username, password, teacher_id),
                )
                conn.commit()
                cur.close()
                conn.close()
                return redirect(url_for("admin_teachers"))
            except Exception as e:
                print("Error updating teacher:", e)
        else:
            cur.execute("SELECT * FROM teachers WHERE teacher_id = %s", (teacher_id,))
            teacher = cur.fetchone()
            cur.close()
            conn.close()
            if teacher:
                return render_template(
                    "update_teacher.html", teacher=teacher, teacher_id=teacher_id
                )
            else:
                return "Teacher not found"
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/delete_teacher/<int:teacher_id>", methods=["POST"])
def delete_teacher(teacher_id):
    if "admin_logged_in" in session:
        try:
            conn = get_db()
            cur = conn.cursor()

            cur.execute(
                "DELETE FROM teacherstudentrelationship WHERE teacher_id = %s",
                (teacher_id,),
            )
            cur.execute("DELETE FROM teachers WHERE teacher_id = %s", (teacher_id,))
            conn.commit()
            cur.close()
            conn.close()
            return redirect(url_for("admin_teachers"))
        except Exception as e:
            flash("An error occurred while deleting the teacher.")
            print(e)
            return redirect(url_for("admin_teachers"))
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/view_teacher_tests/<int:teacher_id>")
def view_teacher_tests(teacher_id):
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM tests WHERE teacher_id = %s", (teacher_id,))
        tests = cur.fetchall()
        cur.close()
        conn.close()
        return render_template("view_teacher_tests.html", tests=tests, teacher_id=teacher_id)
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/view_test_questions/<int:test_id>")
def view_test_questions(test_id):
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM questions WHERE test_id = %s", (test_id,))
        questions = cur.fetchall()

        question_answers = {}
        for question in questions:
            cur.execute(
                "SELECT * FROM expectedanswers WHERE question_id = %s",
                (question[0],),
            )
            answers = cur.fetchall()
            question_answers[question[0]] = answers

        cur.close()
        conn.close()
        return render_template(
            "view_test_questions.html",
            teacher_id=test_id,
            questions=questions,
            question_answers=question_answers,
        )
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/view_question_answers/<int:question_id>")
def view_question_answers(question_id):
    if "admin_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM expectedanswers WHERE question_id = %s", (question_id,)
        )
        answers = cur.fetchall()
        cur.close()
        conn.close()
        return render_template("view_question_answers.html", answers=answers)
    else:
        return redirect(url_for("admin_login"))

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))

# ---------------------- TEACHER ROUTES ---------------------- #

@app.route("/teacher_login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM teachers WHERE username = %s AND password = %s",
            (username, password),
        )
        teacher = cur.fetchone()
        cur.close()
        conn.close()

        if teacher:
            session["teacher_logged_in"] = True
            session["teacher_id"] = teacher[0]
            return redirect(url_for("teacher_home"))
        else:
            return render_template(
                "teacher_login.html", error="Invalid username or password"
            )

    return render_template("teacher_login.html")

@app.route("/teacher_home", methods=["GET", "POST"])
def teacher_home():
    if "teacher_logged_in" in session:
        teacher_id = session["teacher_id"]

        if request.method == "POST":
            conn = get_db()
            cur = conn.cursor()

            if "add_test_name" in request.form:
                test_name = request.form["test_name"]
                cur.execute(
                    "INSERT INTO tests (test_name, teacher_id) VALUES (%s, %s)",
                    (test_name, teacher_id),
                )
                conn.commit()

            elif "update_test_name" in request.form:
                test_id = request.form["test_id"]
                updated_test_name = request.form["updated_test_name"]
                cur.execute(
                    "UPDATE tests SET test_name = %s WHERE test_id = %s",
                    (updated_test_name, test_id),
                )
                conn.commit()

            elif "delete_test_name" in request.form:
                test_id = request.form["test_id"]
                try:
                    cur.execute(
                        "DELETE FROM studentanswers WHERE test_id = %s", (test_id,)
                    )
                    conn.commit()

                    cur.execute(
                        """
                        DELETE FROM expectedanswers 
                        WHERE question_id IN (SELECT question_id FROM questions WHERE test_id = %s)
                        """,
                        (test_id,),
                    )
                    conn.commit()

                    cur.execute("DELETE FROM questions WHERE test_id = %s", (test_id,))
                    conn.commit()

                    cur.execute("DELETE FROM tests WHERE test_id = %s", (test_id,))
                    conn.commit()

                except Exception as e:
                    print("Error:", e)

            cur.close()
            conn.close()

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM tests WHERE teacher_id = %s", (teacher_id,))
        tests = cur.fetchall()
        cur.close()
        conn.close()
        return render_template("teacher_home.html", tests=tests)
    else:
        return redirect(url_for("teacher_login"))

@app.route("/teacher_logout")
def teacher_logout():
    session.pop("teacher_logged_in", None)
    session.pop("teacher_id", None)
    return redirect(url_for("teacher_login"))

@app.route("/teacher/view_test_questions/<int:test_id>", methods=["GET", "POST"])
def view_teacher_test_questions(test_id):
    if "teacher_logged_in" in session:
        conn = get_db()
        cur = conn.cursor()

        if request.method == "POST":
            if "add_question" in request.form:
                question_text = request.form["question_text"]
                expected_answers = request.form.getlist("expected_answer")

                cur.execute(
                    "INSERT INTO questions (question_text, test_id) VALUES (%s, %s) RETURNING question_id",
                    (question_text, test_id),
                )
                question_id = cur.fetchone()[0]

                for answer in expected_answers:
                    cur.execute(
                        "INSERT INTO expectedanswers (answer_text, question_id) VALUES (%s, %s)",
                        (answer, question_id),
                    )

                conn.commit()

            elif "delete_question" in request.form:
                question_id = request.form["question_id"]
                cur.execute(
                    "DELETE FROM expectedanswers WHERE question_id = %s",
                    (question_id,),
                )
                cur.execute(
                    "DELETE FROM questions WHERE question_id = %s", (question_id,)
                )
                conn.commit()

        cur.execute("SELECT * FROM questions WHERE test_id = %s", (test_id,))
        questions = cur.fetchall()

        question_answers = {}
        for question in questions:
            cur.execute(
                "SELECT * FROM expectedanswers WHERE question_id = %s",
                (question[0],),
            )
            answers = cur.fetchall()
            question_answers[question[0]] = answers

        cur.close()
        conn.close()
        return render_template(
            "view_teacher_test_questions.html",
            teacher_id=test_id,
            questions=questions,
            question_answers=question_answers,
        )
    else:
        return redirect(url_for("teacher_login"))


@app.route("/teacher_view_score")
def teacher_view_score():
    if "teacher_logged_in" in session:
        teacher_id = session["teacher_id"]

        conn = get_db()
        cur = conn.cursor()
        query = """
            SELECT s.student_id, s.username AS student_username, t.test_name, 
                   q.question_text, ea.answer_text AS expected_answer, 
                   sa.answer_text AS student_answer, sa.score
            FROM studentanswers sa
            JOIN students s ON sa.student_id = s.student_id
            JOIN tests t ON sa.test_id = t.test_id
            JOIN questions q ON sa.question_id = q.question_id
            JOIN expectedanswers ea ON q.question_id = ea.question_id
            WHERE t.teacher_id = %s
        """
        cur.execute(query, (teacher_id,))
        results = cur.fetchall()
        cur.close()
        conn.close()

        # Use numeric totals; default score -> 0
        student_scores = defaultdict(lambda: {"student_username": None, "tests": defaultdict(list)})
        for result in results:
            student_id, student_username, test_name, question_text, expected_answer, student_answer, score = result

            # Normalize None -> 0
            if score is None:
                score = 0
            # ensure numeric type (in case DB returns Decimal or str)
            try:
                score = int(score)
            except Exception:
                # fallback: convert float-ish values or set 0
                try:
                    score = int(float(score))
                except Exception:
                    score = 0

            student_scores[student_id]["student_username"] = student_username
            student_scores[student_id]["tests"][test_name].append(
                {
                    "question_text": question_text,
                    "expected_answer": expected_answer,
                    "student_answer": student_answer,
                    "score": score,
                }
            )

        # Optionally pre-compute per-test totals so templates don't need to add
        for sid, sdata in student_scores.items():
            for test_name, qlist in sdata["tests"].items():
                total = sum(item.get("score", 0) or 0 for item in qlist)
                max_score = 10 * len(qlist)  # if each question is out of 10
                sdata["tests"][test_name] = {
                    "questions": qlist,
                    "total_score": total,
                    "max_score": max_score,
                    "total_display": f"{total} / {max_score}",
                }

        return render_template("teacher_view_score.html", student_scores=student_scores)
    else:
        return redirect(url_for("teacher_login"))

# ---------------------- STUDENT ROUTES ---------------------- #

def check_test_taken(student_id):
    """Returns True if student has already submitted any answers (simple global check)."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM studentanswers WHERE student_id = %s LIMIT 1",
        (student_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row is not None

@app.route("/student_login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM students WHERE username = %s AND password = %s",
            (username, password),
        )
        student = cur.fetchone()
        cur.close()
        conn.close()

        if student:
            session["student_logged_in"] = True
            session["student_id"] = student[0]
            return redirect(url_for("student_home"))
        else:
            return render_template(
                "student_login.html", error="Invalid username or password"
            )

    return render_template("student_login.html")

@app.route("/student_home")
def student_home():
    if "student_logged_in" in session:
        return render_template("student_home.html")
    else:
        return redirect(url_for("student_login"))

@app.route("/student_logout")
def student_logout():
    session.pop("student_logged_in", None)
    session.pop("student_id", None)
    return redirect(url_for("student_login"))

@app.route("/student_take_test", methods=["GET", "POST"])
def student_take_test():
    if "student_logged_in" in session:
        student_id = session["student_id"]

        if request.method == "POST":
            test_id = request.form.get("test_id")

            if check_test_taken(student_id):
                return redirect(url_for("student_view_score"))

            for key, answer in request.form.items():
                if key.startswith("question_"):
                    question_id = int(key.split("_")[1])

                    conn = get_db()
                    cur = conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO studentanswers (student_id, test_id, question_id, answer_text)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (student_id, test_id, question_id, answer),
                    )
                    conn.commit()
                    cur.close()
                    conn.close()

            return redirect(url_for("student_view_score"))
        else:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT t.test_id, t.test_name 
                FROM tests t 
                LEFT JOIN studentanswers sa 
                       ON t.test_id = sa.test_id AND sa.student_id = %s
                WHERE sa.test_id IS NULL
                """,
                (student_id,),
            )
            tests = cur.fetchall()
            cur.close()
            conn.close()

            tests = [{"test_id": test[0], "test_name": test[1]} for test in tests]
            return render_template("student_take_test.html", tests=tests)
    else:
        return redirect(url_for("student_login"))

@app.route("/student_take_test/<int:test_id>", methods=["GET", "POST"])
def student_take_test_questions(test_id):
    if "student_logged_in" in session:
        student_id = session["student_id"]

        if request.method == "POST":
            for key, answer in request.form.items():
                if key.startswith("question_"):
                    question_id = int(key.split("_")[1])

                    conn = get_db()
                    cur = conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO studentanswers (student_id, test_id, question_id, answer_text)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (student_id, test_id, question_id, answer),
                    )
                    conn.commit()
                    cur.close()
                    conn.close()

            return redirect(url_for("student_home"))
        else:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT * FROM tests WHERE test_id = %s", (test_id,))
            test = cur.fetchone()
            cur.execute("SELECT * FROM questions WHERE test_id = %s", (test_id,))
            questions = cur.fetchall()
            cur.close()
            conn.close()

            return render_template(
                "student_take_test_questions.html",
                test=test,
                questions=questions,
                test_id=test_id,
            )
    else:
        return redirect(url_for("student_login"))

@app.route("/student_view_score")
def student_view_score():
    if "student_logged_in" in session:
        student_id = session["student_id"]

        conn = get_db()
        cur = conn.cursor()
        query = """
            SELECT t.test_id, t.test_name, q.question_text, 
                   ea.answer_text AS expected_answer, 
                   sa.answer_text AS student_answer
            FROM studentanswers sa
            JOIN tests t ON sa.test_id = t.test_id
            JOIN questions q ON sa.question_id = q.question_id
            JOIN expectedanswers ea ON q.question_id = ea.question_id
            WHERE sa.student_id = %s
        """
        cur.execute(query, (student_id,))
        results = cur.fetchall()

        student_scores = {}
        for result in results:
            test_id, test_name, question_text, expected_answer, student_answer = result
            score = evaluate(expected_answer, student_answer)

            cur.execute(
                """
                UPDATE studentanswers 
                SET score = %s 
                WHERE student_id = %s AND test_id = %s 
                  AND question_id IN (SELECT question_id FROM questions WHERE question_text = %s)
                """,
                (score, student_id, test_id, question_text),
            )

            if test_id not in student_scores:
                student_scores[test_id] = {
                    "test_id": test_id,
                    "test_name": test_name,
                    "total_score": 0,
                    "max_score": 0,
                    "scores": [],
                }

            student_scores[test_id]["scores"].append(
                {
                    "question": question_text,
                    "expected_answer": expected_answer,
                    "student_answer": student_answer,
                    "score": score,
                }
            )
            student_scores[test_id]["total_score"] += score
            student_scores[test_id]["max_score"] += 10

        conn.commit()
        cur.close()
        conn.close()

        for test_data in student_scores.values():
            test_data["total_score"] = f"{test_data['total_score']} / {test_data['max_score']}"

        return render_template(
            "student_view_score.html", student_scores=student_scores.values()
        )
    else:
        return redirect(url_for("student_login"))

# ---------------------- MAIN ---------------------- #

if __name__ == "__main__":
    app.run(debug=True)
