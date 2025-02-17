import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# 環境変数の読み込み
load_dotenv()

# OpenAI APIの設定
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
)

# 出力パーサーの設定
output_parser = StrOutputParser()

def create_chain(expert_type: str):
    """
    専門家タイプに応じたチェーンを作成する関数
    
    Args:
        expert_type (str): 専門家タイプ（コーヒー or ワイン）
    
    Returns:
        Chain: 設定されたチェーン
    """
    # 専門家タイプに応じてシステムプロンプトを設定
    if expert_type == "バリスタ(コーヒー)":
        system_prompt = """
        あなたはコーヒーに関する深い知識を持つバリスタです。
        コーヒーの種類、焙煎方法、抽出方法、味わい、香り、産地など、
        コーヒーに関するあらゆる質問に専門的な観点から回答してください。
        """
    else:  # ワインの専門家
        system_prompt = """
        あなたはワインに関する深い知識を持つソムリエです。
        ワインの種類、ブドウ品種、生産地、味わい、香り、保存方法、
        料理とのペアリングなど、ワインに関するあらゆる質問に専門的な観点から回答してください。
        """
    
    # プロンプトテンプレートの作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # チェーンの構築
    chain = prompt | llm | output_parser
    
    return chain

def get_llm_response(prompt: str, expert_type: str) -> str:
    """
    LLMから回答を取得する関数
    
    Args:
        prompt (str): ユーザーからの入力テキスト
        expert_type (str): 専門家タイプ（コーヒー or ワイン）
    
    Returns:
        str: LLMからの回答
    """
    # チェーンの作成
    chain = create_chain(expert_type)
    
    # チェーンの実行
    response = chain.invoke({
        "input": prompt,
        "chat_history": []  # 将来的にチャット履歴を実装する場合に使用
    })
    
    return response

# Streamlitアプリのタイトル設定
st.title("大人な飲み物専門家に質問してみよう！👨‍🍳")

# サイドバーに専門家選択用のラジオボタンを配置
expert_type = st.sidebar.radio(
    "どちらに相談しますか？：",
    ["バリスタ(コーヒー)", "ソムリエ(ワイン)"]
)

# メインエリアに入力フォームを配置
user_input = st.text_area(
    "質問を入力してください：",
    height=100,
    placeholder="例：コーヒーの場合）浅煎りと深煎りの違いは何ですか？\n例：ワインの場合）赤ワインと白ワインの保存方法の違いは？"
)

# 送信ボタン
if st.button("質問する！"):
    if user_input:
        try:
            # ローディング表示
            with st.spinner("回答を生成中..."):
                # LLMから回答を取得
                response = get_llm_response(user_input, expert_type)
                # 回答を表示
                st.write("### 回答：")
                st.write(response)
        except Exception as e:
            st.error(f"エラーが発生しました：{str(e)}")
    else:
        st.warning("質問を入力してください。")