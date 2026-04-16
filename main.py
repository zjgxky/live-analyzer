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
from typing import Dict
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

API_KEY    = os.getenv("DASHSCOPE_API_KEY")
BASE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
UPLOAD_DIR = Path("temp_storage")
UPLOAD_DIR.mkdir(exist_ok=True)

tasks: Dict[str, dict] = {}

# 弹幕坐标基于原始视频分辨率
DANMU_CROP = (25, 1200, 600, 340)
WATCH_CROP = (25, 130,  300,  70)
LIKE_CROP  = (710, 1575, 100, 100)

FRAME_WORKERS        = 20
DEDUP_SIMILARITY     = 0.80
DEDUP_WINDOW_SECONDS = 3

# ================= 提示词 =================

PROMPT_GLOBAL = """
你是一位资深的电商直播运营专家和数据分析师。请对上传的直播视频进行深度复盘，按以下 JSON 格式输出。
输出必须且仅包含一个符合 RFC 8259 规范的 JSON 对象，不要任何开场白，如果某项信息未提及填入 "N/A"，禁止编造。

{
  "video_summary": "100-200字全局概述",
  "host_analysis": {
    "style": "主播风格描述",
    "selling_status": "整体销售表现评价",
    "emotional_shifts": "情绪变化路径及临场反应",
    "interaction_frequency": "互动频率及质量评价"
  },
  "section_info": [
    {"start_time": "MM:SS", "end_time": "MM:SS", "title": "一句话章节标题"}
  ],
  "live_highlights": "100-150字，总结做的好的地方",
  "improvement_suggestions": "100-150字，针对性的优化策略"
}
"""

PROMPT_CHAPTER_DETAIL = """
你是一位资深的电商直播运营专家和数据分析师。请对上传的直播视频切片进行深度复盘，按以下 JSON 格式输出。
输出必须且仅包含一个符合 RFC 8259 规范的 JSON 对象，不要任何开场白，如果某项信息未提及填入 "N/A"，禁止编造。

{
  "video_summary": "50-100字的切片概述",
  "products": [
    {
      "product_name": "产品全称",
      "brand": "品牌",
      "timestamp": "开始-结束时间戳",
      "specs": {"color": "颜色", "material": "材质/成分"},
      "marketing_attributes": {
        "style": "产品风格",
        "occasion": "适用场合",
        "function": "核心功能/卖点",
        "visual_elements": "视觉记忆点"
      },
      "sales_logic": {
        "seed_logic": "种草话术",
        "closing_strategy": "逼单策略",
        "deal_price": "直播间到手价",
        "discount_intensity": "优惠力度"
      },
      "presentation_data": {
        "duration_seconds": "讲解时长（秒）",
        "stock_feedback": "库存情况",
        "audience_concerns": "观众疑问点"
      }
    }
  ]
}
"""

PROMPT_DANMU_TEMPLATE = """
你是一个直播弹幕OCR提取助手。请从截图中提取第 {ts} 秒画面里所有可见的完整弹幕条目。

每条弹幕从左到右：[用户等级] → [用户名] → [弹幕内容]
- [用户等级]：彩色背景方框标签，内含数字或文字，位于最左侧
- [用户名]：紧随等级标签的彩色文字（非白色）
- [弹幕内容]：用户名之后的纯白色文字

规则：三个字段均为必填项，任何一项为空的条目必须丢弃。仅提取完整可见的弹幕，被截断的一律丢弃。

只输出一个合法的 JSON 对象：
{{"danmu_list": [ {{"timestamp": {ts}, "user_level": "...", "user_name": "...", "content": "..."}} ]}}
若没有可识别的弹幕，返回：{{"danmu_list": []}}
"""

PROMPT_WATCH = '提取图片中的观看数字。输出 JSON：{"watch_cnt": "..."}，仅数字，忽略单位。没有则返回空字符串。'
PROMPT_LIKE  = '提取图片中爱心下方的数字。输出 JSON：{"like_cnt": "..."}，仅数字，忽略单位。没有则返回空字符串。'


# ================= 工具函数 =================

def get_oss_policy(model_name: str):
    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    params  = {"action": "getPolicy", "model": model_name}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()['data']

def upload_to_oss(policy_data, file_path):
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"
    with open(file_path, 'rb') as f:
        files = {
            'OSSAccessKeyId':         (None, policy_data['oss_access_key_id']),
            'Signature':              (None, policy_data['signature']),
            'policy':                 (None, policy_data['policy']),
            'x-oss-object-acl':       (None, policy_data['x_oss_object_acl']),
            'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
            'key':                    (None, key),
            'success_action_status':  (None, '200'),
            'file':                   (file_name, f)
        }
        requests.post(policy_data['upload_host'], files=files)
    return f"oss://{key}"

def time_str_to_seconds(t: str) -> float:
    parts = t.strip().split(":")
    return int(parts[0]) * 60 + float(parts[1])


# ================= Part 1 =================

def compress_video(input_path: str) -> str:
    output_path = str(UPLOAD_DIR / f"comp_{uuid.uuid4().hex[:8]}.mp4")
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
            {"type": "text",      "text": PROMPT_GLOBAL}
        ]}],
        extra_body={
            "enable_thinking": False,
            "response_format": {"type": "json_object"}
        },
        extra_headers={"X-DashScope-OssResourceResolve": "enable"}
    )
    return json.loads(completion.choices[0].message.content)


# ================= Part 2 =================

def transcribe_audio(video_path: str) -> list:
    mp3_path = str(UPLOAD_DIR / f"audio_{uuid.uuid4().hex[:8]}.mp3")
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vn',
           '-acodec', 'libmp3lame', '-ac', '1', '-ar', '16000', mp3_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    try:
        policy  = get_oss_policy("fun-asr")
        oss_url = upload_to_oss(policy, mp3_path)
        submit_url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type":  "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable"
        }
        payload = {
            "model": "fun-asr",
            "input": {"file_urls": [oss_url]},
            "parameters": {"diarization_enabled": True, "channel_id": [0]}
        }
        res = requests.post(submit_url, headers=headers, json=payload).json()
        task_id    = res['output']['task_id']
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
                        start_ms  = s['begin_time']
                        end_ms    = s['end_time']
                        start_str = f"{start_ms//60000:02d}:{(start_ms%60000)//1000:02d}"
                        end_str   = f"{end_ms//60000:02d}:{(end_ms%60000)//1000:02d}"
                        sentences.append({
                            "start": start_ms,
                            "end":   end_ms,
                            "text":  f"[{start_str} - {end_str}] 说话人 {s.get('speaker_id','0')}: {s['text']}"
                        })
                return sentences
            elif task_status in ['FAILED', 'CANCELED']:
                raise Exception(f"ASR failed: {status_resp}")
            time.sleep(3)
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


# ================= Part 3 =================
# 关键：从原始视频（非压缩）抽帧，保留原始分辨率，坐标才能匹配

class DanmuAnalyzer:
    def __init__(self):
        self._local = threading.local()

    def _get_client(self):
        if not hasattr(self._local, 'client'):
            self._local.client = OpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
                timeout=30.0
            )
        return self._local.client

    def _safe_api_call(self, prompt: str, img_path: str, retries: int = 2) -> dict:
        client = self._get_client()
        for attempt in range(retries + 1):
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                res = client.chat.completions.create(
                    model="qwen3.5-flash",
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
        h_max, w_max = frame.shape[:2]
        tmp_files = {}
        try:
            for key, (x, y, w, h) in [("danmu", DANMU_CROP), ("watch", WATCH_CROP), ("like", LIKE_CROP)]:
                cropped = frame[y:min(y+h, h_max), x:min(x+w, w_max)]
                if cropped.size == 0:
                    tmp_files[key] = None
                    continue
                path = str(UPLOAD_DIR / f"tmp_{key}_{ts}_{uuid.uuid4().hex[:4]}.jpg")
                cv2.imwrite(path, cropped)
                tmp_files[key] = path

            results = {"timestamp": ts, "danmu_list": [], "watch_cnt": "", "like_cnt": ""}
            if tmp_files.get("danmu"):
                r = self._safe_api_call(PROMPT_DANMU_TEMPLATE.format(ts=ts), tmp_files["danmu"])
                results["danmu_list"] = r.get("danmu_list", [])
            if tmp_files.get("watch"):
                r = self._safe_api_call(PROMPT_WATCH, tmp_files["watch"])
                results["watch_cnt"] = str(r.get("watch_cnt", ""))
            if tmp_files.get("like"):
                r = self._safe_api_call(PROMPT_LIKE, tmp_files["like"])
                results["like_cnt"] = str(r.get("like_cnt", ""))
            return results
        finally:
            for p in tmp_files.values():
                if p and os.path.exists(p):
                    os.remove(p)

    def _fix_merged_fields(self, raw_list: list) -> list:
        for item in raw_list:
            user    = str(item.get("user_name", "") or "").strip()
            content = str(item.get("content",   "") or "").strip()
            if bool(user) == bool(content):
                continue
            merged = user if user else content
            if " " not in merged:
                continue
            split_pos         = merged.index(" ")
            item["user_name"] = merged[:split_pos].strip()
            item["content"]   = merged[split_pos:].strip()
        return raw_list

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
            dup = any(
                ts - p["timestamp"] <= DEDUP_WINDOW_SECONDS and
                sim(user, p["user_name"]) >= DEDUP_SIMILARITY and
                sim(content, p["content"]) >= DEDUP_SIMILARITY
                for p in reversed(retained)
                if ts - p["timestamp"] <= DEDUP_WINDOW_SECONDS
            )
            if not dup:
                retained.append({"timestamp": ts, "user_name": user, "content": content})
                result.append(item)
        return result

    def process(self, original_video_path: str) -> list:
        """从原始视频（非压缩版）抽帧"""
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            return []
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps if fps > 0 else 0
        frame_tasks  = []
        for curr_sec in range(int(duration)):
            frame_id = int(curr_sec * fps)
            if frame_id >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame_tasks.append((curr_sec, frame.copy()))
        cap.release()

        all_raw   = []
        stats_map = {}
        with ThreadPoolExecutor(max_workers=FRAME_WORKERS) as executor:
            futures = {
                executor.submit(self._process_one_second, ts, frame): ts
                for ts, frame in frame_tasks
            }
            for future in as_completed(futures):
                try:
                    res = future.result(timeout=60)
                    all_raw.extend(res["danmu_list"])
                    stats_map[res["timestamp"]] = {
                        "watch_cnt": res["watch_cnt"],
                        "like_cnt":  res["like_cnt"]
                    }
                except Exception:
                    pass

        all_raw.sort(key=lambda x: x.get('timestamp', 0))
        all_raw     = self._fix_merged_fields(all_raw)
        final_danmu = self._deduplicate(all_raw)
        for item in final_danmu:
            stat = stats_map.get(item.get("timestamp"), {})
            item["watch_cnt"] = stat.get("watch_cnt", "N/A")
            item["like_cnt"]  = stat.get("like_cnt",  "N/A")
        return final_danmu


# ================= Part 4 =================
# 关键：切片时重新编码（去掉 -c copy），解决 Invalid video file

def clip_and_analyze_section(comp_video: str, sec: dict, index: int) -> dict:
    clip_path = str(UPLOAD_DIR / f"clip_{index}_{uuid.uuid4().hex[:6]}.mp4")
    try:
        cmd = [
            'ffmpeg', '-y', '-i', comp_video,
            '-ss', sec['start_time'], '-to', sec['end_time'],
            '-vcodec', 'libx264', '-crf', '28', '-preset', 'fast',
            '-acodec', 'aac',
            '-movflags', '+faststart',
            clip_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return {"section_index": index, "start_time": sec.get("start_time",""),
                    "end_time": sec.get("end_time",""), "title": sec.get("title",""),
                    "detail": {"error": f"clip failed: {r.stderr[:200]}"}}

        policy  = get_oss_policy("qwen3.6-plus")
        oss_url = upload_to_oss(policy, clip_path)
        client  = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        comp = client.chat.completions.create(
            model="qwen3.6-plus",
            messages=[{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": oss_url}, "fps": 0.5},
                {"type": "text",      "text": PROMPT_CHAPTER_DETAIL}
            ]}],
            extra_body={
                "enable_thinking": False,
                "response_format": {"type": "json_object"}
            },
            extra_headers={"X-DashScope-OssResourceResolve": "enable"}
        )
        return {
            "section_index": index,
            "start_time": sec["start_time"],
            "end_time":   sec["end_time"],
            "title":      sec.get("title", ""),
            "detail":     json.loads(comp.choices[0].message.content)
        }
    except Exception as e:
        return {
            "section_index": index,
            "start_time": sec.get("start_time",""),
            "end_time":   sec.get("end_time",""),
            "title":      sec.get("title",""),
            "detail":     {"error": str(e)}
        }
    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)

def analyze_chapters(comp_video: str, sections: list) -> list:
    results = [None] * len(sections)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(clip_and_analyze_section, comp_video, sec, i): i
            for i, sec in enumerate(sections)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result(timeout=300)
            except Exception as e:
                results[i] = {
                    "section_index": i,
                    "start_time": sections[i].get("start_time",""),
                    "end_time":   sections[i].get("end_time",""),
                    "title":      sections[i].get("title",""),
                    "detail":     {"error": str(e)}
                }
    return [r for r in results if r is not None]


# ================= Pipeline =================

def pipeline(task_id: str, original_path: str):
    comp_video = None
    try:
        tasks[task_id]["step"] = "压缩视频"
        comp_video = compress_video(original_path)

        tasks[task_id]["step"] = "上传视频 & 启动分析"
        policy  = get_oss_policy("qwen3.6-plus")
        oss_url = upload_to_oss(policy, comp_video)

        part1_result = {}
        part2_result = []
        part3_result = []
        errors       = {}

        def do_part1():
            try:
                tasks[task_id]["step"] = "全局视频分析中"
                part1_result.update(run_global_analysis(oss_url))
                tasks[task_id]["global_info"] = part1_result
            except Exception as e:
                errors["part1"] = str(e)

        def do_part2():
            try:
                part2_result.extend(transcribe_audio(comp_video))
            except Exception as e:
                errors["part2"] = str(e)

        def do_part3():
            try:
                # 传入原始视频
                result = DanmuAnalyzer().process(original_path)
                part3_result.extend(result)
            except Exception as e:
                errors["part3"] = str(e)

        t1 = threading.Thread(target=do_part1)
        t2 = threading.Thread(target=do_part2)
        t3 = threading.Thread(target=do_part3)
        t1.start(); t2.start(); t3.start()
        t1.join()

        if "part1" in errors:
            raise Exception(f"Part1 failed: {errors['part1']}")

        sections = part1_result.get("section_info", [])
        t2.join(); t3.join()

        if errors:
            tasks[task_id]["errors"] = errors

        tasks[task_id]["step"] = "整合章节数据"
        transcription_by_section = []
        for sec in sections:
            start_ms = time_str_to_seconds(sec["start_time"]) * 1000
            end_ms   = time_str_to_seconds(sec["end_time"])   * 1000
            chunk = [s["text"] for s in part2_result
                     if start_ms <= s.get("start", 0) < end_ms]
            transcription_by_section.append("\n".join(chunk))
        tasks[task_id]["transcription_by_section"] = transcription_by_section

        danmu_by_section = []
        for sec in sections:
            start_s = time_str_to_seconds(sec["start_time"])
            end_s   = time_str_to_seconds(sec["end_time"])
            chunk = [d for d in part3_result
                     if start_s <= d.get("timestamp", 0) < end_s]
            danmu_by_section.append(chunk)
        tasks[task_id]["danmu_by_section"] = danmu_by_section

        tasks[task_id]["step"] = "章节细节分析中"
        tasks[task_id]["chapters"] = analyze_chapters(comp_video, sections)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["step"]   = "完成"

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"]  = str(e)
    finally:
        if original_path and os.path.exists(original_path):
            os.remove(original_path)
        if comp_video and os.path.exists(comp_video):
            os.remove(comp_video)


# ================= API Routes =================

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