export async function onRequest(context) {
  const { request, env } = context;

  if (request.method !== 'POST') {
    return json({ error: 'Method Not Allowed' }, 405);
  }

  const apiKey = env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    return json({ error: '服务端未配置 DEEPSEEK_API_KEY 环境变量' }, 500);
  }

  let body;
  try {
    body = await request.json();
  } catch (e) {
    return json({ error: '请求体不是合法 JSON' }, 400);
  }

  const { user, objects } = body;
  if (!user) return json({ error: '缺少 user 字段' }, 400);

  const ctxText = (objects && objects.length)
    ? objects.map(o => `${o.dir}${o.dist}米${o.cls}${o.danger ? '(危险)' : ''}`).join('；')
    : '当前视野内未检测到显著障碍物';

  const systemPrompt = `你是"暖阳视界"导盲系统的 AI 路况解说助手，服务于视障行人。
规则：
1. 全程口语化、短句，不超过 40 个字。
2. 基于"当前路况"给建议，不要编造没检测到的东西。
3. 有危险物（标注"危险"）时优先提醒停下或绕行。
4. 不要说"作为 AI""根据图像"这类废话。
5. 直接给结论，不要列点。`;

  const userPrompt = `当前路况：${ctxText}
用户问题：${user}
请直接给出回答。`;

  try {
    const r = await fetch('https://api.deepseek.com/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: 'deepseek-chat',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        temperature: 0.6,
        max_tokens: 200,
        stream: false
      })
    });

    if (!r.ok) {
      const t = await r.text();
      return json({ error: '上游大模型错误: ' + t.slice(0, 200) }, 502);
    }

    const data = await r.json();
    const reply = data.choices?.[0]?.message?.content?.trim() || '抱歉，我没听清';
    return json({ reply }, 200);
  } catch (e) {
    return json({ error: '调用失败: ' + e.message }, 500);
  }
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8' }
  });
}
