import os
import cv2
import json
import base64
import time
import uuid
import requests
import subprocess
import threading
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 配置区域 =================
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
UPLOAD_DIR = Path("temp_storage")
UPLOAD_DIR.mkdir(exist_ok=True)

# 任务状态存储（demo用内存存储，重启会丢失）
tasks: Dict[str, dict] = {}

# ================= 提示词 =================

PROMPT_GLOBAL_ANALYSIS = """
# Role:
你是一位资深的电商直播运营专家和数据分析师，擅长通过视频分析主播的表现、产品逻辑及直播间的转化效率。

# Task:
请对上传的直播视频切片进行深度复盘。你需要精准提取视频中的关键信息，并按照我要求的 JSON 格式进行结构化输出。

# Analysis Dimensions:
-视频摘要：概括直播切片的核心主题及调性。
-主播表现：评估主播的风格（亲和力/专业度/激情）、销售状态及全程的情绪曲线（是否有断档或高潮）。
-章节速览：根据介绍的产品对音视频内容进行章节划分与总结，记录准确的开始与结束时间点。
-直播亮点：复盘吸引人的瞬间、成功的逼单手段或优秀的互动话术。
-改进建议：基于视频中的不足（如节奏拖沓、光线问题、回复不及时等）提出实操性建议。

# Output Format Rules:
1. 输出必须且仅包含一个符合 RFC 8259 规范的 JSON 对象。
2. 不要包含任何多余的开场白或解释文字。
3. section_info 数组中的每个对象代表一个章节。
4. 如果视频中某项信息未提及，请填入 "N/A"；决不允许编造信息。

# Output Format
{
  "video_summary": "100-200字，全局概述",
  "host_analysis": {
    "style": "主播风格描述",
    "selling_status": "整体销售表现评价",
    "emotional_shifts": "情绪变化路径及临场反应",
    "interaction_frequency": "互动频率及质量评价"
  },
  "section_info": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "title": "一句话概括章节标题"
    }
  ],
  "live_highlights": "100-150字，总结做的好的地方",
  "improvement_suggestions": "100-150字，针对性的优化策略"
}

# Important Note:
必须基于上传的直播视频切片回答，所有提取的信息必须源自视频的语音、字幕或画面。如果视频中没有提到某个维度，该字段必须填写 "N/A"，禁止瞎答，禁止编造虚假数字。
"""

PROMPT_CHAPTER_DETAIL = """
# Role:
你是一位资深的电商直播运营专家和数据分析师，擅长通过视频分析主播的表现、产品逻辑及直播间的转化效率。

# Task:
请对上传的直播视频切片进行深度复盘。你需要精准提取视频中的关键信息，并按照我要求的 JSON 格式进行结构化输出。

# Analysis Dimensions:
-视频摘要：概括直播片段的核心主题及调性。
-产品详情：识别视频中出现的每一个独立产品，并详细记录其物理属性和营销逻辑等信息。

# Output Format Rules:
1. 输出必须且仅包含一个符合 RFC 8259 规范的 JSON 对象。
2. 不要包含任何多余的开场白或解释文字。
3. products 数组中的每个对象代表一个产品。
4. 如果视频中某项信息未提及，请填入 "N/A"；决不允许编造信息。

# Output Format
{
  "video_summary": "50-100字的切片概述",
  "products": [
    {
      "product_name": "产品全称",
      "brand": "品牌",
      "timestamp": "开始-结束时间戳",
      "specs": {
        "color": "颜色",
        "material": "材质/成分"
      },
      "marketing_attributes": {
        "style": "产品风格",
        "occasion": "适用场合",
        "function": "核心功能/卖点",
        "visual_elements": "视觉记忆点"
      },
      "sales_logic": {
        "seed_logic": "种草话术/为什么要买",
        "closing_strategy": "逼单策略（如：限时、限额、低价诱惑）",
        "deal_price": "直播间到手价",
        "discount_intensity": "优惠力度计算及评价"
      },
      "presentation_data": {
        "duration_seconds": "讲解时长（秒）",
        "stock_feedback": "视频中反馈的销售/库存情况",
        "audience_concerns": "该产品讲解期间观众最集中的疑问点"
      }
    }
  ]
}

# Note [Important]:
必须基于上传的直播视频切片回答，所有提取的信息必须源自视频的语音、字幕或画面。如果视频中没有提到某个维度，该字段必须填写 "N/A"，禁止瞎答，禁止编造虚假数字。
"""

# 弹幕OCR提示词：注意用 .format(ts=ts) 替换 {ts}
PROMPT_DANMU_OCR_TEMPLATE = """
你是一个直播弹幕OCR提取助手。请从截图中提取第 {ts} 秒画面里所有可见的完整弹幕条目。

【视觉结构说明】：
每条弹幕在画面中从左到右依次由三部分组成：
- [用户等级]：彩色背景的方框标签，内含数字或文字等级，位于该条弹幕的最左侧
- [用户名]：紧随等级标签之后的彩色文字（非白色），与弹幕内容之间有明显间隔
- [弹幕内容]：用户名之后的纯白色文字，直到该行结束

【提取规则】：
1. 必须严格按照 [用户等级] → [用户名] → [弹幕内容] 的顺序提取，不得调换或合并。
2. [用户名] 与 [弹幕内容] 之间以明显空格分隔：空格左侧为用户名（彩色），右侧为内容（白色）。
3. 三个字段均为必填项，任何一项为空的条目必须丢弃，不得输出。
4. 仅提取画面中完整可见的弹幕；被截断、被遮挡的条目一律丢弃。

【输出要求】：
只输出一个合法的 JSON 对象：
{{"danmu_list": [ {{"timestamp": {ts}, "user_level": "...", "user_name": "...", "content": "..."}} ]}}
若画面中没有可识别的弹幕，返回：{{"danmu_list": []}}
"""

PROMPT_WATCH_CNT = '提取图片中的观看数字。必须输出 JSON 对象，格式为：{"watch_cnt": "..."}，仅输出数字本身，忽略计数单位。如果没有找到数字，返回空字符串。'
PROMPT_LIKE_CNT  = '提取图片中爱心下方的数字。必须输出 JSON 对象，格式为：{"like_cnt": "..."}，仅输出数字本身，忽略计数单位。如果没有找到数字，返回空字符串。'


# ================= 工具函数 =================

def get_oss_policy(model_name: str):
    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    params = {"action": "getPolicy", "model": model_name}
    response = requests.get(url, headers=headers, params=params)
    return response.json()['data']

def upload_to_oss(policy_data, file_path):
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"
    with open(file_path, 'rb') as file:
        files = {
            'OSSAccessKeyId':      (None, policy_data['oss_access_key_id']),
            'Signature':           (None, policy_data['signature']),
            'policy':              (None, policy_data['policy']),
            'x-oss-object-acl':    (None, policy_data['x_oss_object_acl']),
            'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
            'key':                 (None, key),
            'success_action_status': (None, '200'),
            'file':                (file_name, file)
        }
        requests.post(policy_data['upload_host'], files=files)
    return f"oss://{key}"

def time_str_to_seconds(t: str) -> float:
    """把 MM:SS 转成秒数"""
    parts = t.strip().split(":")
    return int(parts[0]) * 60 + float(parts[1])


# ================= Part 1：压缩 + 全局分析 =================

def compress_video(input_path: str) -> str:
    output_path = str(UPLOAD_DIR / f"comp_{uuid.uuid4()}.mp4")
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', "scale='if(gt(iw,ih),min(720,iw),-2):if(gt(ih,iw),min(720,ih),-2)'",
        '-vcodec', 'libx264', '-crf', '30', '-preset', 'fast',
        '-acodec', 'aac', '-ac', '1', '-ar', '16000',
        '-movflags', '+faststart', output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path

def run_global_analysis(video_oss_url: str) -> dict:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": video_oss_url}, "fps": 0.2},
            {"type": "text", "text": PROMPT_GLOBAL_ANALYSIS}
        ]}],
        extra_body={
            "enable_thinking": False,
            "response_format": {"type": "json_object"}   # ✅ 放在 extra_body，不是 extra_headers
        },
        extra_headers={"X-DashScope-OssResourceResolve": "enable"}
    )
    raw = completion.choices[0].message.content
    return json.loads(raw)


# ================= Part 2：语音转录 =================

def transcribe_audio(video_path: str) -> list:
    """返回带时间戳的句子列表，每项包含 start(ms), end(ms), text"""
    # 提取单声道音频，加 uuid 防止并发冲突
    mp3_path = str(UPLOAD_DIR / f"audio_{uuid.uuid4()}.mp3")
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'libmp3lame', '-ac', '1', '-ar', '16000', mp3_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    try:
        policy = get_oss_policy("fun-asr")
        oss_url = upload_to_oss(policy, mp3_path)

        submit_url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable"
        }
        payload = {
            "model": "fun-asr",
            "input": {"file_urls": [oss_url]},
            "parameters": {"diarization_enabled": True, "channel_id": [0]}
        }
        res = requests.post(submit_url, headers=headers, json=payload).json()
        task_id = res['output']['task_id']

        status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        while True:
            status_resp = requests.get(
                status_url, headers={"Authorization": f"Bearer {API_KEY}"}
            ).json()
            task_status = status_resp['output']['task_status']
            if task_status == 'SUCCEEDED':
                final_url = status_resp['output']['results'][0]['transcription_url']
                data = requests.get(final_url).json()
                sentences = []
                for tr in data.get('transcripts', []):
                    for s in tr.get('sentences', []):
                        start_ms = s['begin_time']
                        end_ms   = s['end_time']
                        start_str = f"{start_ms//60000:02d}:{(start_ms%60000)//1000:02d}"
                        end_str   = f"{end_ms//60000:02d}:{(end_ms%60000)//1000:02d}"
                        sentences.append({
                            "start": start_ms,
                            "end":   end_ms,
                            "text":  f"[{start_str} - {end_str}] 说话人 {s.get('speaker_id', '0')}: {s['text']}"
                        })
                return sentences
            elif task_status in ['FAILED', 'CANCELED']:
                raise Exception(f"ASR任务失败: {status_resp}")
            time.sleep(3)
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


# ================= Part 3：弹幕提取 =================

# 弹幕区域坐标（根据视频实际分辨率调整）
DANMU_CROP = (25, 1200, 600, 340)   # x, y, w, h
WATCH_CROP = (25, 130,  300,  70)
LIKE_CROP  = (710, 1575, 100, 100)
FRAME_WORKERS = 10  # Railway 基础版降低并发数
DEDUP_SIMILARITY_THRESHOLD = 0.80
DEDUP_WINDOW_SECONDS = 3

class DanmuAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)

    def _api_call(self, prompt: str, img_path: str, retries: int = 2) -> dict:
        for attempt in range(retries + 1):
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                res = self.client.chat.completions.create(
                    model="qwen3-vl-plus",
                    messages=[{"role": "user", "content": [
                        {"type": "text",      "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]}],
                    extra_body={"enable_thinking": False},
                    response_format={"type": "json_object"}
                )
                return json.loads(res.choices[0].message.content)
            except Exception:
                if attempt < retries:
                    time.sleep((attempt + 1) * 2)
        return {}

    def _process_one_second(self, ts: int, frame) -> dict:
        """裁剪、识别单帧，返回该秒的弹幕+统计数据"""
        h_max, w_max = frame.shape[:2]
        tmp_files = {}
        results = {}
        try:
            for key, (x, y, w, h) in [("danmu", DANMU_CROP), ("watch", WATCH_CROP), ("like", LIKE_CROP)]:
                cropped = frame[y:min(y+h, h_max), x:min(x+w, w_max)]
                path = str(UPLOAD_DIR / f"tmp_{key}_{ts}_{uuid.uuid4().hex[:6]}.jpg")
                cv2.imwrite(path, cropped)
                tmp_files[key] = path

            # 弹幕OCR：用 .format(ts=ts) 填入时间戳 ✅
            danmu_res = self._api_call(PROMPT_DANMU_OCR_TEMPLATE.format(ts=ts), tmp_files["danmu"])
            watch_res = self._api_call(PROMPT_WATCH_CNT, tmp_files["watch"])
            like_res  = self._api_call(PROMPT_LIKE_CNT,  tmp_files["like"])

            results = {
                "timestamp":  ts,
                "danmu_list": danmu_res.get("danmu_list", []),
                "watch_cnt":  str(watch_res.get("watch_cnt", "")),
                "like_cnt":   str(like_res.get("like_cnt",  ""))
            }
        finally:
            # ✅ 删除当前帧的临时文件，而不是永远删 [0]
            for p in tmp_files.values():
                if os.path.exists(p):
                    os.remove(p)
        return results

    def _deduplicate(self, raw_list: list) -> list:
        def sim(a, b):
            if not a and not b: return 1.0
            if not a or not b:  return 0.0
            return SequenceMatcher(None, a, b).ratio()

        sorted_list = sorted(raw_list, key=lambda x: x.get("timestamp", 0))
        retained, result = [], []
        for item in sorted_list:
            ts      = item.get("timestamp", 0)
            user    = item.get("user_name", "")
            content = item.get("content", "")
            dup = False
            for prev in reversed(retained):
                if ts - prev["timestamp"] > DEDUP_WINDOW_SECONDS: break
                if sim(user, prev["user_name"]) >= DEDUP_SIMILARITY_THRESHOLD and \
                   sim(content, prev["content"]) >= DEDUP_SIMILARITY_THRESHOLD:
                    dup = True
                    break
            if not dup:
                retained.append({"timestamp": ts, "user_name": user, "content": content})
                result.append(item)
        return result

    def process(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_tasks  = []

        curr_sec = 0
        while True:
            frame_id = int(curr_sec * fps)
            if frame_id >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                break
            frame_tasks.append((curr_sec, frame.copy()))
            curr_sec += 1
        cap.release()

        all_danmu_raw = []
        stats_map     = {}

        with ThreadPoolExecutor(max_workers=FRAME_WORKERS) as executor:
            futures = {
                executor.submit(self._process_one_second, ts, frame): ts
                for ts, frame in frame_tasks
            }
            for future in as_completed(futures):
                try:
                    res = future.result(timeout=60)
                    all_danmu_raw.extend(res["danmu_list"])
                    stats_map[res["timestamp"]] = {
                        "watch_cnt": res["watch_cnt"],
                        "like_cnt":  res["like_cnt"]
                    }
                except Exception:
                    pass

        all_danmu_raw.sort(key=lambda x: x.get('timestamp', 0))
        final_list = self._deduplicate(all_danmu_raw)

        # 把对应秒的统计数据合并进来
        for item in final_list:
            stat = stats_map.get(item.get("timestamp"), {})
            item["watch_cnt"] = stat.get("watch_cnt", "N/A")
            item["like_cnt"]  = stat.get("like_cnt",  "N/A")

        return final_list


# ================= Part 4：章节细节分析 =================

def analyze_chapters(video_path: str, sections: list) -> list:
    """逐章节切片 → 上传 → 调用大模型分析"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = []

    for i, sec in enumerate(sections):
        clip_path = str(UPLOAD_DIR / f"clip_{uuid.uuid4()}.mp4")
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', video_path,
                 '-ss', sec['start_time'], '-to', sec['end_time'],
                 '-c', 'copy', clip_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )

            policy  = get_oss_policy("qwen3.6-plus")
            oss_url = upload_to_oss(policy, clip_path)

            comp = client.chat.completions.create(
                model="qwen3.6-plus",
                messages=[{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": oss_url}, "fps": 0.5},
                    {"type": "text",      "text": PROMPT_CHAPTER_DETAIL}
                ]}],
                extra_body={
                    "enable_thinking": False,
                    "response_format": {"type": "json_object"}   # ✅ 从 extra_headers 移到 extra_body
                },
                extra_headers={"X-DashScope-OssResourceResolve": "enable"}
            )
            results.append({
                "section_index": i,
                "start_time": sec["start_time"],
                "end_time":   sec["end_time"],
                "title":      sec.get("title", ""),
                "detail":     json.loads(comp.choices[0].message.content)
            })
        except Exception as e:
            results.append({
                "section_index": i,
                "start_time": sec["start_time"],
                "end_time":   sec["end_time"],
                "title":      sec.get("title", ""),
                "detail":     {"error": str(e)}
            })
        finally:
            if os.path.exists(clip_path):
                os.remove(clip_path)

    return results


# ================= 核心工作流（并发版） =================

def _filter_by_section(items: list, section: dict, key_ms: str = "start") -> list:
    """按章节时间戳过滤列表（items 里每条有毫秒级时间戳 start/end）"""
    sec_start = time_str_to_seconds(section["start_time"]) * 1000
    sec_end   = time_str_to_seconds(section["end_time"])   * 1000
    return [i for i in items if sec_start <= i.get(key_ms, 0) < sec_end]

def _filter_danmu_by_section(danmu_list: list, section: dict) -> list:
    """弹幕的时间戳是秒，不是毫秒"""
    sec_start = time_str_to_seconds(section["start_time"])
    sec_end   = time_str_to_seconds(section["end_time"])
    return [d for d in danmu_list if sec_start <= d.get("timestamp", 0) < sec_end]

def pipeline(task_id: str, file_path: str):
    try:
        # ── 步骤1：压缩视频 ──────────────────────────────────
        tasks[task_id]["step"] = "压缩视频"
        comp_video = compress_video(file_path)
        os.remove(file_path)  # 原始上传文件不再需要

        # ── 步骤2：上传到OSS（qwen模型用），并发启动 Part1/3 ──
        tasks[task_id]["step"] = "上传视频 & 启动分析"
        policy  = get_oss_policy("qwen3.6-plus")
        oss_url = upload_to_oss(policy, comp_video)

        # Part1（全局分析）和 Part3（弹幕）并发跑
        part1_result = {}
        part3_result = []
        part1_error  = []

        def do_part1():
            try:
                part1_result.update(run_global_analysis(oss_url))
                tasks[task_id]["global_info"] = part1_result
                tasks[task_id]["step"] = "全局分析完成"
            except Exception as e:
                part1_error.append(str(e))

        def do_part3():
            try:
                tasks[task_id]["step"] = "弹幕提取中"
                result = DanmuAnalyzer().process(comp_video)
                part3_result.extend(result)
                # 弹幕不在 status 接口里直接返回全量，太大了
                # 切分后才写入 tasks
            except Exception as e:
                tasks[task_id]["danmu_error"] = str(e)

        t1 = threading.Thread(target=do_part1)
        t3 = threading.Thread(target=do_part3)
        t1.start()
        t3.start()

        # Part2（ASR转录）也同时启动
        asr_result = []
        def do_part2():
            try:
                tasks[task_id]["step"] = "语音转录中"
                result = transcribe_audio(comp_video)
                asr_result.extend(result)
            except Exception as e:
                tasks[task_id]["asr_error"] = str(e)

        t2 = threading.Thread(target=do_part2)
        t2.start()

        # 等 Part1 完成再继续（需要章节时间戳）
        t1.join()
        if part1_error:
            raise Exception(f"全局分析失败: {part1_error[0]}")

        sections = part1_result.get("section_info", [])

        # 等 Part2/3 完成，再按章节切分
        t2.join()
        t3.join()

        # 按章节切分转录文本
        tasks[task_id]["step"] = "整合章节数据"
        transcription_by_section = []
        for sec in sections:
            sec_start_ms = time_str_to_seconds(sec["start_time"]) * 1000
            sec_end_ms   = time_str_to_seconds(sec["end_time"])   * 1000
            chunk = [
                s["text"] for s in asr_result
                if sec_start_ms <= s.get("start", 0) < sec_end_ms
            ]
            transcription_by_section.append("\n".join(chunk))
        tasks[task_id]["transcription_by_section"] = transcription_by_section

        # 按章节切分弹幕
        danmu_by_section = []
        for sec in sections:
            chunk = _filter_danmu_by_section(part3_result, sec)
            danmu_by_section.append(chunk)
        tasks[task_id]["danmu_by_section"] = danmu_by_section

        # ── 步骤4：章节细节分析（Part4）──────────────────────
        tasks[task_id]["step"] = "章节细节分析"
        chapters = analyze_chapters(comp_video, sections)
        tasks[task_id]["chapters"] = chapters

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["step"]   = "完成"

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"]  = str(e)
    finally:
        if 'comp_video' in locals() and os.path.exists(comp_video):
            os.remove(comp_video)


# ================= API 路由 =================

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id    = str(uuid.uuid4())
    local_path = str(UPLOAD_DIR / f"{task_id}_{file.filename}")

    with open(local_path, "wb") as f:
        f.write(await file.read())

    tasks[task_id] = {
        "status": "processing",
        "step":   "已上传",
        "global_info":               {},
        "transcription_by_section":  [],
        "danmu_by_section":          [],
        "chapters":                  []
    }
    background_tasks.add_task(pipeline, task_id, local_path)
    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
