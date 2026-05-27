import streamlit as st
import pickle
import string
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

nltk.download("stopwords",quiet=True)

st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_files():
    model=load_model("mental_health_rnn_model.keras")

    with open("tokenizer.pkl","rb") as f:
        tokenizer=pickle.load(f)

    with open("label_encoder.pkl","rb") as f:
        label_encoder=pickle.load(f)

    return model,tokenizer,label_encoder

model,tokenizer,label_encoder=load_files()

max_len=80
stop_words=set(stopwords.words("english"))

risk_rules={
    "Suicidal":[
        "want to die","kill myself","suicide","end my life",
        "no reason to live","hurt myself","self harm",
        "i want to disappear","i do not want to live"
    ],
    "Depression":[
        "hopeless","worthless","depressed","feeling empty",
        "nothing matters","very sad","crying everyday",
        "lost interest","life is meaningless"
    ],
    "Anxiety":[
        "panic attack","anxious","overthinking","cannot sleep",
        "cant sleep","heart racing","feeling nervous",
        "i am scared","worried all the time"
    ],
    "Stress":[
        "stressed","stress","too much pressure","mentally exhausted",
        "burned out","overloaded","work pressure","tired of everything"
    ],
    "Bipolar":[
        "mood swings","extreme happiness","extreme sadness",
        "suddenly energetic","suddenly depressed","high and low mood"
    ],
    "Personality disorder":[
        "identity crisis","unstable relationships",
        "emotional instability","fear of abandonment",
        "impulsive behavior"
    ],
    "Normal":[
        "happy","calm","peaceful","doing well",
        "everything is fine","excited","good day"
    ]
}

suggestions={
    "Suicidal":[
        "Immediate counselor or guardian attention is recommended.",
        "Do not leave the person alone if this is a real case.",
        "Encourage contacting emergency support or a trusted person."
    ],
    "Depression":[
        "Encourage the user to talk to a counselor or trusted person.",
        "Suggest simple routines like sleep, food, and light activity.",
        "Monitor repeated negative messages over time."
    ],
    "Anxiety":[
        "Suggest deep breathing or grounding exercises.",
        "Encourage reducing immediate stress triggers.",
        "Recommend counselor support if anxiety continues."
    ],
    "Stress":[
        "Suggest taking a short break.",
        "Encourage task prioritization and rest.",
        "Recommend support if stress is continuous."
    ],
    "Bipolar":[
        "Suggest professional mental health evaluation.",
        "Monitor sudden emotional changes.",
        "Encourage maintaining a mood diary."
    ],
    "Personality disorder":[
        "Suggest counselor or therapist support.",
        "Encourage emotional regulation activities.",
        "Monitor relationship or identity-related distress."
    ],
    "Normal":[
        "Emotional state appears stable.",
        "Encourage maintaining healthy habits.",
        "No immediate concern detected."
    ]
}

def clean_text(text):
    text=str(text).lower()
    text=text.translate(str.maketrans("","",string.punctuation))
    words=text.split()
    words=[word for word in words if word not in stop_words]
    return " ".join(words)

def predict_sentiment(sentence):
    original=sentence.lower()

    for label,phrases in risk_rules.items():
        for phrase in phrases:
            if phrase in original:
                return label,"Rule-Based Alert",100,True

    cleaned=clean_text(sentence)
    seq=tokenizer.texts_to_sequences([cleaned])
    pad=pad_sequences(seq,maxlen=max_len,padding="post",truncating="post")

    pred=model.predict(pad,verbose=0)
    class_index=np.argmax(pred)
    confidence=float(np.max(pred)*100)
    label=label_encoder.inverse_transform([class_index])[0]

    return label,round(confidence,2),round(confidence),False

def risk_level(label,confidence):
    if label=="Suicidal":
        return "Critical"
    elif label in ["Depression","Anxiety","Stress","Bipolar","Personality disorder"]:
        if confidence=="Rule-Based Alert" or confidence>=70:
            return "High"
        else:
            return "Moderate"
    else:
        return "Low"

if "history" not in st.session_state:
    st.session_state.history=[]

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")
st.write("This app analyzes user text and predicts emotional sentiment using a Simple RNN model with rule-based safety monitoring.")

col1,col2=st.columns([2,1])

with col1:
    user_text=st.text_area(
        "Enter user message",
        height=180,
        placeholder="Example: I feel stressed and anxious today..."
    )

    realtime=st.checkbox("Enable real-time analysis",value=True)

    analyze_button=st.button("Analyze Sentiment")

with col2:
    st.info("Classes supported: Normal, Depression, Anxiety, Stress, Suicidal, Bipolar, Personality disorder")

run_prediction=False

if realtime and user_text.strip()!="":
    run_prediction=True

if analyze_button:
    run_prediction=True

if run_prediction:
    if user_text.strip()=="":
        st.warning("Please enter some text.")
    else:
        label,confidence,progress,is_rule=predict_sentiment(user_text)
        level=risk_level(label,confidence)

        st.divider()
        st.subheader("Prediction Result")

        c1,c2,c3=st.columns(3)

        c1.metric("Predicted Sentiment",label)
        c2.metric("Confidence",str(confidence))
        c3.metric("Risk Level",level)

        if confidence!="Rule-Based Alert":
            st.progress(progress/100)

        if label=="Normal":
            st.success("Emotional Status: Stable or positive emotional pattern detected.")
        elif label=="Suicidal":
            st.error("Critical Alert: High-risk suicidal expression detected. Immediate counselor or trusted-person intervention is recommended.")
        else:
            st.warning("Negative emotional trend detected. Counselor review may be helpful.")

        st.subheader("Suggestions")

        for item in suggestions.get(label,[]):
            st.write("•",item)

        st.subheader("Message Details")

        cleaned=clean_text(user_text)

        st.write("**Original Text:**",user_text)
        st.write("**Cleaned Text:**",cleaned)

        if is_rule:
            st.write("**Detection Type:** Rule-based safety detection")
        else:
            st.write("**Detection Type:** RNN model prediction")

        st.session_state.history.append({
            "Text":user_text,
            "Prediction":label,
            "Confidence":confidence,
            "Risk Level":level
        })

st.divider()

st.subheader("Prediction History")

if len(st.session_state.history)>0:
    st.dataframe(st.session_state.history[::-1],use_container_width=True)
else:
    st.write("No predictions yet.")

st.divider()

st.caption("Disclaimer: This app is for academic demonstration only and should not replace professional mental health diagnosis or emergency support.")