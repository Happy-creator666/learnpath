"""
学习计划生成器 - Flask 后端
提供对话管理、DeepSeek API 调用、B站课程搜索
"""
import json
import os
import re
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# ==================== CONFIG ====================
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise RuntimeError('DEEPSEEK_API_KEY environment variable is not set')
DEEPSEEK_API_URL = 'https://api.deepseek.com/chat/completions'
DEEPSEEK_MODEL = 'deepseek-chat'

# ==================== STATE MACHINE ====================
STATES = {
    'INIT': 'init',
    'ASK_BASIS': 'ask_basis',
    'ASK_HOURS': 'ask_hours',
    'ASK_GOAL': 'ask_goal',
    'ASK_STYLE': 'ask_style',
    'PLAN_READY': 'plan_ready',
}

# 用户会话存储（生产环境应换为 Redis/DB）
sessions = {}


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            'state': STATES['INIT'],
            'profile': {'skill': '', 'basis': '', 'hours': '', 'goal': '', 'style': ''},
            'plan_weeks': [],
            'plan_raw': ''
        }
    return sessions[session_id]


# ==================== DEEPSEEK API ====================
def call_deepseek(messages, temperature=0.7, max_tokens=4096):
    """调用 DeepSeek API"""
    resp = requests.post(
        DEEPSEEK_API_URL,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
        },
        json={
            'model': DEEPSEEK_MODEL,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        },
        timeout=60
    )
    if not resp.ok:
        err = resp.json() if resp.text else {}
        raise Exception(err.get('error', {}).get('message', f'HTTP {resp.status_code}'))
    return resp.json()['choices'][0]['message']['content']


# ==================== B站 API ====================
def search_bilibili(keyword, page=1, page_size=10):
    """搜索B站视频，按播放量排序，筛选长视频（课程类）"""
    url = 'https://api.bilibili.com/x/web-interface/search/type'
    params = {
        'search_type': 'video',
        'keyword': keyword,
        'page': page,
        'order': 'click',  # 按播放量排序
        'duration': 4,     # 30-60分钟以上的长视频（课程特征）
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.bilibili.com/'
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if data.get('code') != 0:
            return []

        results = []
        for v in data.get('data', {}).get('result', []):
            play = v.get('play', 0)
            # 格式化播放量
            if play >= 10000:
                play_str = f'{play/10000:.1f}万'
            else:
                play_str = str(play)

            # 格式化时长
            duration = v.get('duration', '')
            if duration:
                parts = duration.split(':')
                if len(parts) == 2:
                    dur_min = int(parts[0])
                    dur_sec = int(parts[1])
                    dur_str = f'{dur_min}:{dur_sec:02d}'
                else:
                    dur_str = duration
            else:
                dur_str = '--:--'

            results.append({
                'bvid': v.get('bvid', ''),
                'title': v.get('title', '').replace('<em class="keyword">', '').replace('</em>', ''),
                'author': v.get('author', ''),
                'play': play,
                'play_str': play_str,
                'duration': dur_str,
                'url': v.get('arcurl', f'https://www.bilibili.com/video/{v.get("bvid", "")}'),
                'pic': v.get('pic', ''),
                'description': v.get('description', ''),
                'tag': v.get('tag', ''),
            })

        # 按播放量降序排
        results.sort(key=lambda x: x['play'], reverse=True)
        return results[:page_size]

    except Exception as e:
        print(f'B站搜索失败: {e}')
        return []


def search_bilibili_playlist(keyword):
    """搜索B站合集/系列课程"""
    # 系列课程通常在标题中含有特定关键词
    playlist_keywords = [
        f'{keyword} 全套教程',
        f'{keyword} 入门到精通',
        f'{keyword} 零基础',
        f'{keyword} 完整版',
        f'{keyword} 系统课程',
    ]
    all_results = []
    seen = set()
    for kw in playlist_keywords[:3]:  # 限制3个搜索词
        results = search_bilibili(kw, page=1, page_size=5)
        for r in results:
            if r['bvid'] not in seen:
                seen.add(r['bvid'])
                all_results.append(r)

    all_results.sort(key=lambda x: x['play'], reverse=True)
    return all_results[:8]


# ==================== PLAN GENERATION ====================
def generate_plan(profile):
    """调用 DeepSeek 生成学习计划"""
    system_prompt = """你是一个顶级学习规划师。根据用户背景生成详细学习计划。

## 输出格式（必须严格遵守）：
每周用以下JSON格式输出（方便前端解析）：

```json
{
  "weeks": [
    {
      "week": 1,
      "title": "周标题",
      "tasks": [
        {"text": "具体学习任务1（要详细，含具体技术/知识点）"},
        {"text": "具体学习任务2"}
      ],
      "bilibili_search": "B站搜索关键词1, 关键词2"
    }
  ],
  "summary": "计划概述和鼓励语"
}
```

## 要求：
- 根据基础调整难度（零基础从安装配置开始）
- 根据每周时间决定任务量（<5h→2-3个任务, 5-10h→3-4个, >10h→4-6个）
- 根据偏好侧重推荐资源类型
- 生成4-8周计划
- bilibili_search 字段给出精准的B站搜索关键词，用于搜索高播放量课程"""

    user_prompt = f"""
我想学：{profile['skill']}
当前基础：{profile['basis']}
每周时间：{profile['hours']}
学习目标：{profile['goal']}
学习偏好：{profile['style']}

请生成个性化的周学习计划。"""

    response = call_deepseek([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ])

    # 尝试解析 JSON
    try:
        # 提取 JSON 部分
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            plan_data = json.loads(json_match.group(1))
        else:
            # 尝试直接解析
            plan_data = json.loads(response)

        # 为每周搜索B站课程
        for week in plan_data.get('weeks', []):
            keyword = week.get('bilibili_search', f"{profile['skill']} {week.get('title', '')}")
            week['resources'] = search_bilibili_playlist(keyword)

        return plan_data

    except (json.JSONDecodeError, KeyError):
        # 如果AI没有返回有效JSON，回退到文本解析
        return {'weeks': [], 'summary': response, 'raw': True}


def adjust_plan(profile, plan_weeks, feedback):
    """根据用户反馈调整计划"""
    messages = [
        {'role': 'system', 'content': '你是学习规划师。根据用户反馈调整计划，保持JSON格式输出（同 generate 格式），bilibili_search 字段更新。'},
        {'role': 'user', 'content': f"""原计划：{json.dumps(plan_weeks, ensure_ascii=False)}
用户反馈：{feedback}

输出调整后的完整计划JSON（同格式）。"""}
    ]
    response = call_deepseek(messages)
    try:
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        plan_data = json.loads(json_match.group(1)) if json_match else json.loads(response)
        for week in plan_data.get('weeks', []):
            keyword = week.get('bilibili_search', f"{profile['skill']} {week.get('title', '')}")
            week['resources'] = search_bilibili_playlist(keyword)
        return plan_data
    except:
        return {'weeks': [], 'summary': response, 'raw': True}


# ==================== API ROUTES ====================

@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    核心对话接口
    请求: { session_id, message }
    响应: { reply, state, chips, plan_data? }
    """
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    user_msg = data.get('message', '').strip()

    if not user_msg:
        return jsonify({'error': '消息不能为空'}), 400

    sess = get_session(session_id)
    state = sess['state']
    profile = sess['profile']
    reply = ''
    chips = []
    next_state = state

    try:
        # ---- INIT: 等待技能输入 ----
        if state == STATES['INIT']:
            profile['skill'] = user_msg
            next_state = STATES['ASK_BASIS']
            reply = f'好的！你想学习「<b>{user_msg}</b>」。\n\n首先，请问你目前有相关基础吗？'
            chips = ['零基础，完全新手', '有一些基础，了解基本概念', '中级水平，想系统提升', '高级，想深入特定方向']

        # ---- ASK_BASIS ----
        elif state == STATES['ASK_BASIS']:
            profile['basis'] = user_msg
            next_state = STATES['ASK_HOURS']
            reply = '了解了。你每周大概能投入多少时间来学习？'
            chips = ['不到5小时，碎片时间', '5-10小时，每天1-2小时', '10-20小时，比较充裕', '20小时以上，全力冲刺']

        # ---- ASK_HOURS ----
        elif state == STATES['ASK_HOURS']:
            profile['hours'] = user_msg
            next_state = STATES['ASK_GOAL']
            reply = '明白。你的学习目标是什么？想达到什么水平？'
            chips = ['入门了解，能看懂就行', '独立完成小项目', '达到求职/实习水平', '考取相关证书']

        # ---- ASK_GOAL ----
        elif state == STATES['ASK_GOAL']:
            profile['goal'] = user_msg
            next_state = STATES['ASK_STYLE']
            reply = '最后一个问题：你更喜欢哪种学习方式？'
            chips = ['看视频教程为主', '读文档/书籍为主', '动手做项目为主', '都可以，你推荐就好']

        # ---- ASK_STYLE -> 生成计划 ----
        elif state == STATES['ASK_STYLE']:
            profile['style'] = user_msg
            next_state = STATES['PLAN_READY']

            # 生成计划
            plan_data = generate_plan(profile)
            sess['plan_weeks'] = plan_data.get('weeks', [])
            sess['plan_raw'] = plan_data.get('summary', '')

            reply = f"""✅ 你的专属学习计划已生成！

📝 <b>{profile['skill']}</b> | 📊 {profile['basis']} | ⏰ {profile['hours']} | 🎯 {profile['goal']}

右侧面板展示了完整的周计划，每周围绕具体任务推荐了B站高播放量课程，点击即可观看。"""
            chips = ['💾 保存并新建学习项目', '📺 推荐更多资源', '✏️ 调整当前计划']

        # ---- PLAN_READY ----
        elif state == STATES['PLAN_READY']:
            if any(kw in user_msg for kw in ['新', '重新', '重来', '开始']):
                sess['state'] = STATES['INIT']
                sess['profile'] = {'skill': '', 'basis': '', 'hours': '', 'goal': '', 'style': ''}
                sess['plan_weeks'] = []
                reply = '好的，让我们重新开始！你想学习什么技能？'
                chips = []
                return jsonify({'reply': reply, 'state': sess['state'], 'chips': chips, 'plan_weeks': []})

            elif any(kw in user_msg for kw in ['资源', '推荐']):
                extra = recommend_extra(profile)
                reply = extra
                chips = ['💾 保存并新建学习项目', '📺 推荐更多资源', '✏️ 调整当前计划', '💡 提问学习问题']
                return jsonify({
                    'reply': reply, 'state': state, 'chips': chips,
                    'plan_weeks': sess.get('plan_weeks', []), 'plan_changed': False
                })

            elif any(kw in user_msg for kw in ['调整', '修改', '变更', '增加', '减少', '调一下', '改一下']):
                plan_data = adjust_plan(profile, sess.get('plan_weeks', []), user_msg)
                sess['plan_weeks'] = plan_data.get('weeks', [])
                reply = '✅ 计划已根据你的反馈更新，查看右侧面板获取最新内容。'
                chips = ['💾 保存并新建学习项目', '📺 推荐更多资源', '✏️ 调整当前计划', '💡 提问学习问题']

            else:
                reply = tutor_answer(profile, sess.get('plan_weeks', []), user_msg)
                chips = ['💾 保存并新建学习项目', '📺 推荐更多资源', '✏️ 调整当前计划', '💡 继续提问']
                return jsonify({
                    'reply': reply, 'state': state, 'chips': chips,
                    'plan_weeks': sess.get('plan_weeks', []), 'plan_changed': False
                })

        # ---- 更新状态 ----
        sess['state'] = next_state

        return jsonify({
            'reply': reply,
            'state': next_state,
            'chips': chips,
            'plan_weeks': sess.get('plan_weeks', []),
            'profile': profile
        })

    except Exception as e:
        return jsonify({'error': str(e), 'reply': f'❌ 出错了：{str(e)}'}), 500


@app.route('/api/search-bilibili', methods=['POST'])
def bilibili_search():
    """
    B站课程搜索接口
    请求: { keyword }
    响应: { results: [...] }
    """
    data = request.get_json()
    keyword = data.get('keyword', '')
    if not keyword:
        return jsonify({'error': '关键词不能为空'}), 400

    results = search_bilibili_playlist(keyword)
    return jsonify({'results': results})


@app.route('/api/adjust-plan', methods=['POST'])
def adjust():
    """
    根据完成度调整计划
    请求: { session_id, feedback }
    """
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    feedback = data.get('feedback', '')

    sess = get_session(session_id)
    plan_data = adjust_plan(sess['profile'], sess.get('plan_weeks', []), feedback)
    sess['plan_weeks'] = plan_data.get('weeks', [])

    return jsonify({
        'reply': '✅ 计划已调整！',
        'plan_weeks': sess.get('plan_weeks', [])
    })


def recommend_extra(profile):
    """调用AI生成额外资源推荐"""
    try:
        resp = call_deepseek([
            {'role': 'system', 'content': '推荐5-8个优质学习资源。包含B站关键词、YouTube频道、GitHub项目、知名教程网站。格式简洁，每条一行，带分类标签。'},
            {'role': 'user', 'content': f"我在学{profile['skill']}，基础{profile['basis']}。请推荐更多免费资源。"}
        ])
        return resp
    except:
        return '📺 试试在B站搜索：' + profile['skill'] + ' 教程'


def tutor_answer(profile, plan_weeks, question):
    """学习答疑：根据用户的学习计划和问题，用DeepSeek进行解答"""
    # 提取当前学习计划的摘要上下文
    plan_context = ''
    if plan_weeks:
        plan_context = '当前学习计划：\n'
        for w in plan_weeks[:6]:
            plan_context += f"第{w.get('week','?')}周「{w.get('title','')}」: "
            tasks = [t.get('text', '') for t in (w.get('tasks', []) or [])]
            plan_context += '、'.join(tasks[:4]) + '\n'

    try:
        resp = call_deepseek([
            {'role': 'system', 'content': f"""你是一位耐心、专业的编程学习导师。学生在学「{profile.get('skill', '未知技能')}」，基础水平：{profile.get('basis', '未知')}，学习目标：{profile.get('goal', '未知')}。

你的回答风格：
- 用通俗易懂的语言解释，配合代码示例（如果适用）
- 如果学生问的是概念/原理，先给一句话总结，再展开详解
- 如果学生问的是调试/报错，分析原因并提供解决方案
- 如果学生问的是学习方法，结合他们的基础和目标给出建议
- 适当鼓励，营造积极的学习氛围
- 回答控制在200-600字，不要太长
- 使用中文回答，技术术语保留英文原文"""},
            {'role': 'user', 'content': f"""{plan_context}
学生提问：{question}

请以导师身份耐心解答。"""}
        ], temperature=0.7, max_tokens=2048)
        return resp
    except Exception as e:
        return f'抱歉，答疑服务暂时不可用：{str(e)}。请稍后重试。'


@app.route('/api/restore', methods=['POST'])
def restore():
    """
    恢复会话状态（页面刷新后同步前端 localStorage 到后端 session）
    请求: { session_id, profile, plan_weeks }
    """
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    profile = data.get('profile', {})
    plan_weeks = data.get('plan_weeks', [])
    sessions[session_id] = {
        'state': STATES['PLAN_READY'],
        'profile': profile,
        'plan_weeks': plan_weeks,
        'plan_raw': ''
    }
    return jsonify({'ok': True, 'state': STATES['PLAN_READY']})


@app.route('/api/reset', methods=['POST'])
def reset():
    """重置会话"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    sessions[session_id] = {
        'state': STATES['INIT'],
        'profile': {'skill': '', 'basis': '', 'hours': '', 'goal': '', 'style': ''},
        'plan_weeks': [],
        'plan_raw': ''
    }
    return jsonify({'ok': True})


@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz():
    """
    生成练习题
    请求: { skill, week_title, tasks: [{text: "..."}] }
    响应: { questions: [{q, options: [A,B,C,D], answer: 0-3}] }
    """
    data = request.get_json()
    skill = data.get('skill', '')
    week_title = data.get('week_title', '')
    tasks = data.get('tasks', [])
    preference = data.get('preference', '')

    if not skill or not tasks:
        return jsonify({'error': '参数不完整'}), 400

    tasks_text = '\n'.join([f"- {t['text']}" for t in tasks])

    extra_req = ''
    if preference:
        extra_req = f'\n用户的额外要求：{preference}'

    try:
        resp = call_deepseek([
            {'role': 'system', 'content': f"""你是教育专家，根据学习内容生成高质量单选题。

返回纯JSON（不要markdown包裹）：
{{"questions": [
  {{"q": "题目", "options": ["A选项", "B选项", "C选项", "D选项"], "answer": 0}}
]}}

要求：
- 默认生成5道题，如果用户要求更多则按用户要求
- 4个选项，answer为正确选项的索引(0-3)
- 难度递进，覆盖核心知识点
- 选项要有迷惑性，不能一眼看出答案
- 如果用户要求特定题型（代码题、概念题等），按用户要求调整
- 题目语言：中文""",
            },
            {'role': 'user', 'content': f"技能：{skill}\n章节：{week_title}\n学习内容：\n{tasks_text}{extra_req}\n\n请生成题目。"}
        ], temperature=0.8, max_tokens=4096)

        json_match = re.search(r'\{[\s\S]*\}', resp)
        if json_match:
            quiz_data = json.loads(json_match.group(0))
        else:
            quiz_data = json.loads(resp)

        return jsonify({'questions': quiz_data.get('questions', [])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-summary', methods=['POST'])
def generate_summary():
    """
    生成每周知识摘要
    请求: { skill, week_title, tasks: [{text: "..."}], resources: [...] }
    响应: { summary: "markdown格式的知识摘要" }
    """
    data = request.get_json()
    skill = data.get('skill', '')
    week_title = data.get('week_title', '')
    tasks = data.get('tasks', [])

    if not skill or not tasks:
        return jsonify({'error': '参数不完整'}), 400

    tasks_text = '\n'.join([f"- {t['text']}" for t in tasks])

    try:
        resp = call_deepseek([
            {'role': 'system', 'content': """你是知识提炼专家。根据学习内容生成一份详尽的知识摘要（800-1500字）。

格式要求（Markdown）：
### 核心概念
- 列出3-5个最重要的概念，每个用2-3句话详细解释，尽可能附带简短的代码示例

### 关键知识点
- 列出5-8个必须掌握的知识点，用通俗语言详细解释每个知识点，说明其重要性和应用场景
- **每个知识点必须附带一个可运行的代码示例**，用 ``` 代码块包裹，代码要简洁实用

### 常见误区
- 列出2-3个初学者最容易犯的错误或混淆点，每个误区附带错误代码 vs 正确代码的对比示例

### 进阶提示
- 给2-3条进阶学习建议，附带代码演示帮助学习者深入理解

### 推荐练习
- 给出2-3个具体的练习思路，每个练习配一个代码框架或提示

语言：中文，深入浅出。代码示例务必简洁、正确、可直接运行。""",
            },
            {'role': 'user', 'content': f"技能：{skill}\n章节：{week_title}\n学习内容：\n{tasks_text}\n\n请生成详细的知识摘要。"}
        ], temperature=0.5, max_tokens=2048)

        return jsonify({'summary': resp.strip()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    print('LearnPath backend starting...')
    print('Visit: http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
