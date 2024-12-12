import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader
import joblib
import numpy as np
from textblob import TextBlob
from pythainlp.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sqlalchemy import create_engine

def load_data_from_db():
    """โหลดข้อมูลจากฐานข้อมูล MySQL และส่งคืนเป็น DataFrame"""
    try:
        engine = create_engine('mysql+mysqlconnector://root:1234@localhost/reviewapp')
        
        query_content = "SELECT * FROM contentbasedview;"
        content_based_data = pd.read_sql(query_content, con=engine)
        print("โหลดข้อมูล Content-Based สำเร็จ")
        
        query_collaborative = "SELECT * FROM collaborativeview;"
        collaborative_data = pd.read_sql(query_collaborative, con=engine)
        print("โหลดข้อมูล Collaborative สำเร็จ")
        
        return content_based_data, collaborative_data
    except Exception as e:
        print(f"ข้อผิดพลาดในการโหลดข้อมูลจากฐานข้อมูล: {str(e)}")
        raise

def normalize_scores(series):
    """ทำให้คะแนนอยู่ในช่วง [0, 1]"""
    min_val, max_val = series.min(), series.max()
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return series

def normalize_engagement(data, user_column='owner_id', engagement_column='PostEngagement'):
    """ปรับ Engagement ให้เหมาะสมตามผู้ใช้แต่ละคนให้อยู่ในช่วง [0, 1]"""
    data['NormalizedEngagement'] = data.groupby(user_column)[engagement_column].transform(lambda x: normalize_scores(x))
    return data

def analyze_comments(comments):
    """วิเคราะห์ความรู้สึกของคอมเมนต์ รองรับทั้งภาษาไทยและภาษาอังกฤษ"""
    sentiment_scores = []
    for comment in comments:
        try:
            if pd.isna(comment):
                sentiment_scores.append(0)
            else:
                # หากเป็นภาษาไทย ให้ tokenize ด้วย PyThaiNLP
                if any('\u0E00' <= char <= '\u0E7F' for char in comment):
                    tokenized_comment = ' '.join(word_tokenize(comment, engine='newmm'))
                else:
                    tokenized_comment = comment

                # คำนวณ Sentiment ด้วย TextBlob
                blob = TextBlob(tokenized_comment)
                polarity = blob.sentiment.polarity
                
                # กำหนด Sentiment Score
                if polarity > 0.5:
                    sentiment_scores.append(1)  # Sentiment บวก
                elif 0 < polarity <= 0.5:
                    sentiment_scores.append(1)  # Sentiment บวก
                elif -0.5 <= polarity < 0:
                    sentiment_scores.append(-1)  # Sentiment ลบ
                else:
                    sentiment_scores.append(-1)  # Sentiment ลบ
        except Exception as e:
            sentiment_scores.append(0)  # หากเกิดข้อผิดพลาด ให้คะแนนเป็น 0
    return sentiment_scores

def create_collaborative_model(data, n_factors=150, n_epochs=70, lr_all=0.008, reg_all=0.1):
    """สร้างและฝึกโมเดล Collaborative Filtering"""
    required_columns = ['user_id', 'post_id']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"ข้อมูลขาดคอลัมน์ที่จำเป็น: {set(required_columns) - set(data.columns)}")

    melted_data = data.melt(id_vars=['user_id', 'post_id'], var_name='category', value_name='score')
    melted_data = melted_data[melted_data['score'] > 0]

    reader = Reader(rating_scale=(melted_data['score'].min(), melted_data['score'].max()))
    interaction_data = Dataset.load_from_df(melted_data[['user_id', 'post_id', 'score']], reader)

    trainset = interaction_data.build_full_trainset()
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    model.fit(trainset)

    joblib.dump(model, 'Collaborative_Model.pkl')
    return model

def create_content_based_model(data, text_column='Content', comment_column='Comments', engagement_column='PostEngagement'):
    """สร้างโมเดล Content-Based Filtering ด้วย TF-IDF และ KNN"""
    required_columns = [text_column, comment_column, engagement_column]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"ข้อมูลขาดคอลัมน์ที่จำเป็น: {set(required_columns) - set(data.columns)}")

    # ใช้ TF-IDF เพื่อแปลงเนื้อหาของโพสต์เป็นเวกเตอร์
    tfidf = TfidfVectorizer(stop_words='english', max_features=6000, ngram_range=(1, 3), min_df=1, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(data[text_column].fillna(''))

    # ใช้ KNN เพื่อหาความคล้ายคลึงระหว่างโพสต์
    knn = NearestNeighbors(n_neighbors=30, metric='cosine')
    knn.fit(tfidf_matrix)

    # วิเคราะห์ความรู้สึกจากความคิดเห็น
    data['SentimentScore'] = analyze_comments(data[comment_column])
    data = normalize_engagement(data)

    # คำนวณคะแนนการมีส่วนร่วมที่ปรับแล้ว (ใช้การบวกคะแนน Sentiment)
    data['NormalizedEngagement'] = normalize_scores(data[engagement_column])  # Normalize Engagement ให้อยู่ในช่วง [0, 1]
    data['WeightedEngagement'] = data['NormalizedEngagement'] + data['SentimentScore']  # บวกค่า Engagement กับ Sentiment Score

    joblib.dump(tfidf, 'TFIDF_Model.pkl')
    joblib.dump(knn, 'KNN_Model.pkl')
    return tfidf, knn, data

def recommend_hybrid(user_id, data, collaborative_model, knn, categories, tfidf, alpha=0.50):
    """แนะนำโพสต์โดยใช้ Hybrid Filtering รวม Collaborative และ Content-Based โดยคำนึงถึงหมวดหมู่"""
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha ต้องอยู่ในช่วง 0 ถึง 1")

    # ขั้นแรก: หาข้อมูลโพสต์ที่ผู้ใช้เคยโต้ตอบแล้ว
    interacted_posts = data[data['owner_id'] == user_id]['post_id'].tolist()

    # ข้อมูลโพสต์ที่ยังไม่ได้ดู
    unviewed_data = data[~data['post_id'].isin(interacted_posts)]

    recommendations = []

    # ขั้นที่สอง: ใช้หมวดหมู่ในการเลือกโพสต์ที่แนะนำ
    for category in categories:
        category_data = unviewed_data[unviewed_data[category] == 1]

        # ถ้าไม่มีโพสต์ในหมวดหมู่นั้น ๆ ให้ข้ามไป
        if category_data.empty:
            continue
        
        for _, post in category_data.iterrows():
            # Collaborative Filtering: คำนวณคะแนนจากโมเดล Collaborative
            collab_score = collaborative_model.predict(user_id, post['post_id']).est

            # Content-Based Filtering: คำนวณคะแนนจากความคล้ายคลึงของเนื้อหา
            idx = data.index[data['post_id'] == post['post_id']].tolist()
            content_score = 0
            if idx:
                idx = idx[0]
                # แปลงเนื้อหาของโพสต์เป็นเวกเตอร์ TF-IDF
                tfidf_vector = tfidf.transform([data.iloc[idx]['Content']])
                
                # ใช้ KNN เพื่อหาความคล้ายคลึงของโพสต์
                n_neighbors = min(30, knn._fit_X.shape[0])  # ใช้ shape[0] แทน len()
                distances, indices = knn.kneighbors(tfidf_vector, n_neighbors=n_neighbors)  # หาความคล้ายของโพสต์
                
                # คำนวณคะแนนจากโพสต์ที่คล้ายกัน
                content_score = np.mean([data.iloc[i]['NormalizedEngagement'] for i in indices[0]])

            # ผสมคะแนนจาก Collaborative และ Content-Based ตามค่า alpha
            final_score = alpha * collab_score + (1 - alpha) * content_score
            recommendations.append((post['post_id'], final_score))

    # จัดเรียงโพสต์ตามคะแนนที่ได้และคำนวณคะแนนที่เป็น normalized score
    recommendations_df = pd.DataFrame(recommendations, columns=['post_id', 'score'])
    recommendations_df['normalized_score'] = normalize_scores(recommendations_df['score'])
    recommendations = recommendations_df.sort_values(by='normalized_score', ascending=False)['post_id'].tolist()

    return recommendations



def evaluate_model(data, recommendations, threshold=0.5):
    """ประเมินผลโมเดลด้วย Precision, Recall, F1-Score และ Accuracy"""
    relevant_items = data[(data['PostEngagement'] > threshold) & (data['SentimentScore'] > -1)]['post_id'].tolist()
    recommended_items = recommendations

    tp = set(recommended_items) & set(relevant_items)
    fp = set(recommended_items) - tp
    fn = set(relevant_items) - tp


    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # คำนวณ Accuracy
    accuracy = len(tp) / len(recommended_items) if len(recommended_items) > 0 else 0

    return precision, recall, f1, accuracy, list(tp), list(fp), list(fn)


def plot_evaluation_results(results):
    """วาดกราฟผลการประเมิน Precision, Recall, F1 และ Accuracy"""
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
    averages = [
        np.mean([r[0] for r in results]),
        np.mean([r[1] for r in results]),
        np.mean([r[2] for r in results]),
        np.mean([r[3] for r in results])
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, averages, color=['blue', 'green', 'red', 'purple'])
    plt.ylim(0, 1)
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.show()
    print("กราฟผลการประเมินถูกบันทึกใน 'evaluation_metrics.png'")


def plot_confusion_matrix(tp, fp, fn):
    """วาดกราฟ Confusion Matrix"""
    cm = np.array([[len(tp), len(fp)], [len(fn), len(tp)]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
    plt.title('Confusion Matrix for Recommendation System')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    print("Confusion Matrix ถูกบันทึกใน 'confusion_matrix.png'")

def main():
    # โหลดข้อมูลจากฐานข้อมูล
    content_based_data, collaborative_data = load_data_from_db()

    if collaborative_data is None or content_based_data is None:
        return

    # สร้างโมเดล Collaborative Filtering
    collaborative_model = create_collaborative_model(collaborative_data)

    # สร้างโมเดล Content-Based Filtering
    tfidf, knn, enriched_content_data = create_content_based_model(content_based_data)

    # หมวดหมู่ตัวอย่าง
    categories = [
        'Gadget', 'Smartphone', 'Laptop', 'Smartwatch', 'Headphone', 'Tablet', 'Camera', 'Drone',
        'Home_Appliance', 'Gaming_Console', 'Wearable_Device', 'Fitness_Tracker', 'VR_Headset',
        'Smart_Home', 'Power_Bank', 'Bluetooth_Speaker', 'Action_Camera', 'E_Reader',
        'Desktop_Computer', 'Projector'
    ]

    user_ids = enriched_content_data['owner_id'].unique()
    results = []
    all_tp, all_fp, all_fn = [], [], []

    for _ in range(5):  # รันซ้ำ 5 รอบเพื่อความเสถียร
        for user_id in user_ids:
            recommendations = recommend_hybrid(user_id, enriched_content_data, collaborative_model, knn, categories, tfidf, alpha=0.5)
            precision, recall, f1, accuracy, tp, fp, fn = evaluate_model(enriched_content_data, recommendations)
            results.append((precision, recall, f1, accuracy))
            all_tp.extend(tp)
            all_fp.extend(fp)
            all_fn.extend(fn)

    # คำนวณค่าเฉลี่ยของผลการประเมิน
    avg_precision = np.mean([r[0] for r in results])
    avg_recall = np.mean([r[1] for r in results])
    avg_f1 = np.mean([r[2] for r in results])
    avg_accuracy = np.mean([r[3] for r in results])  # ค่าเฉลี่ย Accuracy

    print("ผลการประเมินเฉลี่ยหลังจาก 5 รอบ:")
    print(f"Precision: {avg_precision:.2f}")
    print(f"Recall: {avg_recall:.2f}")
    print(f"F1 Score: {avg_f1:.2f}")
    print(f"Accuracy: {avg_accuracy:.2f}")  # แสดง Accuracy

    # วาดกราฟผลการประเมิน
    plot_evaluation_results(results)
    plot_confusion_matrix(all_tp, all_fp, all_fn)

    # บันทึกผลการประเมินลงไฟล์ CSV
    results_df = pd.DataFrame(results, columns=['Precision', 'Recall', 'F1', 'Accuracy'])  # เพิ่ม Accuracy
    results_df.to_csv('evaluation_results.csv', index=False)
    print("ผลการประเมินถูกบันทึกใน 'evaluation_results.csv'")

    # แสดงผลการแนะนำสำหรับ user_id 1170001
    test_user_id = 1170001
    test_recommendations = recommend_hybrid(test_user_id, enriched_content_data, collaborative_model, knn, categories, tfidf, alpha=0.5)
    print(f"โพสต์ที่แนะนำสำหรับ user_id {test_user_id}: {test_recommendations[:10]}")

if __name__ == "__main__":
    main()