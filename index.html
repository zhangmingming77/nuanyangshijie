<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>暖阳视界 - 盲道路况 AI 播报</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#111;color:#fff;font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei"}
  #status{padding:10px;font-size:16px;text-align:center;background:#1e293b}
  .btn-row{display:flex;gap:8px;padding:8px;background:#0f172a}
  button{flex:1;padding:16px;font-size:18px;background:#2563eb;color:white;border:none;border-radius:8px;font-weight:600}
  button.secondary{background:#059669}
  button.listening{background:#dc2626;animation:pulse 1s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
  #wrap{position:relative}
  video{width:100%;display:block;object-fit:cover;background:#000}
  canvas{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
  #log{height:200px;overflow-y:auto;padding:10px;font-size:14px;background:#1f2937;white-space:pre-wrap;line-height:1.6}
  #aiTip{padding:8px;font-size:13px;color:#fbbf24;background:#422006;border-top:1px solid #78350f;min-height:32px}
</style>
</head>
<body>
<div id="status">点击下方按钮开启</div>
<div class="btn-row">
  <button id="btnStart">🔊 开启播报</button>
  <button id="btnAsk" class="secondary">🎙 我要问路</button>
</div>
<div id="aiTip">AI 智能解说：开启后将由大模型结合当前路况生成安全建议</div>
<div id="wrap">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>
</div>
<div id="log">— 暖阳视界 启动中 —</div>

<script>
/* ========== 基础配置 ========== */
const targetClasses = ["person","bicycle","car","motorcycle","bus","truck","fire hydrant","traffic light","stop sign","bench","potted plant","dog","cat"];
const CLASS_CN = {person:"行人",bicycle:"自行车",car:"汽车",motorcycle:"摩托车",bus:"公交车",truck:"卡车","fire hydrant":"消防栓","traffic light":"红绿灯","stop sign":"停车标志",bench:"长椅","parking meter":"停车计时器","potted plant":"盆栽",dog:"狗",cat:"猫"};
const KNOWN = {person:1.7,bicycle:1.0,car:1.5,motorcycle:1.2,bus:3,truck:3,"fire hydrant":0.9,"traffic light":2.5,"stop sign":1,bench:0.8,"parking meter":1.2,"potted plant":0.6,dog:0.5,cat:0.3};
const VFOV = 50 * Math.PI / 180;
const DANGER = 3;
const COOLDOWN = 3500;
const AI_INTERVAL = 12000;

const video=document.getElementById('video');
const canvas=document.getElementById('canvas');
const ctx=canvas.getContext('2d');
const logDiv=document.getElementById('log');
const statusDiv=document.getElementById('status');
const aiTip=document.getElementById('aiTip');
const btn=document.getElementById('btnStart');
const btnAsk=document.getElementById('btnAsk');

let model=null;
let track=new Map();
let isVoice=false;
let lastObjects=[];
let lastAiTime=0;
let running=false;

/* ========== 语音播报队列 ========== */
class VQ{
  constructor(){this.q=[];this.p=false;}
  speak(t){this.q.push(t);if(!this.p)this.go();}
  go(){
    if(!this.q.length){this.p=false;return;}
    this.p=true;
    const u=new SpeechSynthesisUtterance(this.q.shift());
    u.lang='zh-CN'; u.rate=1.0; u.pitch=1.0;
    u.onend=()=>{setTimeout(()=>this.go(),150)};
    speechSynthesis.speak(u);
  }
}
const vq=new VQ();

function log(m){
  const d=new Date();
  logDiv.innerText+=`[${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}] ${m}\n`;
  logDiv.scrollTop=logDiv.scrollHeight;
}

/* ========== 检测辅助 ========== */
function clockDir(nx){
  const dx=(nx-0.5)*2;
  const a=dx*90;
  let c=Math.round(a/30);
  if(c===0)return 12;
  if(c>0)return c;
  return 12+c;
}
function getDist(cls,h,vh){
  const known=KNOWN[cls]||1;
  const fp=(vh/2)/Math.tan(VFOV/2);
  if(h<=0)return 99;
  return (known*fp)/h;
}

/* ========== 调用大模型（走自家后端代理，key 不暴露） ========== */
async function callLLM(userText, contextObjects){
  try{
    const r=await fetch('/api/llm',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({user:userText, objects:contextObjects})
    });
    if(!r.ok) throw new Error('HTTP '+r.status);
    const data=await r.json();
    return data.reply;
  }catch(e){
    log('AI 调用失败:'+e.message);
    return null;
  }
}

/* ========== 主检测循环 ========== */
async function loop(){
  if(!running) return;
  if(!model||!video.srcObject){ setTimeout(loop,300); return; }
  try{
    const preds=await model.detect(video,30,0.4,0.5);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const vw=video.videoWidth, vh=video.videoHeight;
    canvas.width=vw; canvas.height=vh;
    const now=Date.now();
    const objs=[];

    for(const o of preds){
      const cls=o.class;
      if(targetClasses.indexOf(cls)<0) continue;
      const b=o.bbox, x=b[0], y=b[1], w=b[2], h=b[3];
      const nx=(x+w/2)/vw, ny=(y+h/2)/vh;
      const dist=getDist(cls,h,vh);
      const danger=dist<DANGER;
      const dir=clockDir(nx);
      const clockName=dir===12?'正前方':dir+'点钟方向';
      const cn=CLASS_CN[cls]||cls;

      ctx.strokeStyle=danger?'#ff3333':'#ffdd00';
      ctx.lineWidth=3; ctx.strokeRect(x,y,w,h);
      ctx.font='16px sans-serif';
      ctx.fillStyle=danger?'#ff3333':'#ffdd00';
      ctx.fillText(`${cn} ${dist.toFixed(1)}m`,x,y-6);

      objs.push({cls:cn, dist:dist.toFixed(1), dir:clockName, danger});

      const id=cls+'_'+Math.round(nx*10)+'_'+Math.round(ny*10);
      if(now-(track.get(id)||0)>=COOLDOWN){
        track.set(id,now);
        const prefix=danger?'危险！':'';
        const txt=`${prefix}${clockName}，${dist.toFixed(1)}米，${cn}`;
        vq.speak(txt);
        log((danger?'⚠️ ':'📍 ')+txt);
      }
    }
    lastObjects=objs;
    for(const [k,t] of track){ if(now-t>COOLDOWN*3) track.delete(k); }

    // 每 12 秒，把当前路况交给大模型做一次综合安全解说
    if(now-lastAiTime>=AI_INTERVAL && objs.length>0){
      lastAiTime=now;
      const aiReply=await callLLM('请用一句话、口语化、30字以内，结合当前路况给视障行人一个安全提醒。', objs);
      if(aiReply){
        aiTip.textContent='🤖 AI：'+aiReply;
        vq.speak(aiReply);
        log('🤖 AI 解说：'+aiReply);
      }
    }
  }catch(e){ log('识别出错:'+e.message); }
  setTimeout(loop,600);
}

/* ========== 开启按钮 ========== */
btn.onclick=async()=>{
  if(!isVoice){
    vq.speak('语音已开启，暖阳视界开始播报');
    isVoice=true; running=true;
  }
  statusDiv.innerText='正在开启摄像头…';
  try{
    if(!video.srcObject){
      const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
      video.srcObject=stream;
    }
    statusDiv.innerText='正在加载 AI 模型…';
    model=await cocoSsd.load();
    statusDiv.innerText='✅ 运行中：识别盲道物体并播报（含大模型安全解说）';
    log('✅ 模型加载完成，盲道模式 + AI 解说已启用');
    loop();
  }catch(e){
    statusDiv.innerText='出错了:'+e.message;
    log('错误:'+e.message);
  }
};

/* ========== 语音问答按钮 ========== */
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
let sr=null, asking=false;
if(SR){
  sr=new SR();
  sr.lang='zh-CN'; sr.interimResults=false; sr.maxAlternatives=1;
  sr.onresult=async(e)=>{
    const q=e.results[0][0].transcript;
    log('🎙 你问：'+q);
    vq.speak('你问的是，'+q);
    btnAsk.classList.remove('listening');
    btnAsk.textContent='🎙 我要问路';
    const reply=await callLLM(q, lastObjects);
    if(reply){
      aiTip.textContent='🤖 AI：'+reply;
      vq.speak(reply);
      log('🤖 AI 答：'+reply);
    }else{
      vq.speak('网络或大模型暂时不可用，请稍后再试');
    }
    asking=false;
  };
  sr.onerror=(e)=>{ log('语音识别错误:'+e.error); asking=false; btnAsk.classList.remove('listening'); btnAsk.textContent='🎙 我要问路'; };
}
btnAsk.onclick=()=>{
  if(!SR){ alert('当前浏览器不支持语音识别，请用 Chrome/Edge'); return; }
  if(asking){ sr.stop(); asking=false; btnAsk.classList.remove('listening'); btnAsk.textContent='🎙 我要问路'; return; }
  if(!running){ alert('请先点击"开启播报"，让摄像头工作起来'); return; }
  asking=true;
  btnAsk.classList.add('listening');
  btnAsk.textContent='🛑 聆听中…再点结束';
  vq.speak('请说你的问题，比如，前面能走吗？');
  sr.start();
};
</script>
</body>
</html>
