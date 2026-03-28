# 粗筛失败处理流程改进方案

## 项目背景

当用户搜索丢失物品时，如果粗筛（coarse screening）失败，系统当前显示固定的提示文字：
- "Not a match — please provide more details"
- "Describe the item clearly (category, color, material, distinguishing features) and include the location where you lost it."

这种固定提示不够个性化，不能针对用户具体输入的缺陷进行指导。

## 改进目标

通过调用外部LLM，对用户的输入文本进行**智能特征识别和分析**，诊断文本中：
- ✅ 已包含的特征类型
- ⚠️ 缺失的关键特征类型
- 💡 针对性的补充建议

替代固定提示，提供**个性化的特征补充指导**。

---

## 可直接交给AI的开发任务书

将下面整段作为实现指令给AI即可：

```text
你正在修改 Lost-Article-Matching-Platform 项目。请按以下约束实现，不要偏离。

[业务规则]
1) 只在粗筛失败时调用特征诊断LLM。
2) 用户最多补充2次：
     - assist_round=0/1：返回诊断建议。
     - assist_round>=2 且仍无匹配：直接返回终态固定文案，不再调用LLM。
3) 终态固定文案：
     Item is not currently in our database. We will notify you by email if a matching record appears.
4) 不实现邮件发送逻辑。
5) 数据库变更只允许“新增字段”，禁止删除或修改已有字段。
6) 为防止数据库膨胀：两次失败后执行数据瘦身与延迟清理。
7) LLM可识别多语言输入，但返回建议必须是英文。
8) 缺失特征提示优先级：先提示“物品属性类”(category/color/material/brand/size)。

[需要修改的文件]
- app/search/routes.py
- app/search/diagnostic.py (new)
- templates/search/match_failed.html
- service/models/... (RequestModel对应模型文件，仅新增字段)

[接口契约]
POST /search/screen 新增入参：assist_round(int, default=0)

返回（有匹配）：
{
    "screening": {...},
    "terminal": {"reached": false},
    "outcome": {...}
}

返回（无匹配，且assist_round<2）：
{
    "screening": {...},
    "diagnostic": {
        "identified_features": {...},
        "missing_attributes": [...],
        "missing_features": [...],
        "suggestions": ["English only", "..."]
    },
    "terminal": {"reached": false},
    "assist_round_next": assist_round + 1,
    "outcome": {...}
}

返回（无匹配，且assist_round>=2）：
{
    "screening": {...},
    "diagnostic": null,
    "terminal": {
        "reached": true,
        "message": "Item is not currently in our database. We will notify you by email if a matching record appears."
    },
    "outcome": {...}
}

[数据清理规则]
当 assist_round>=2 且无匹配：
1) 立即将 diagnostic_data / identified_features / missing_features 置空；
2) terminal_reached=true, terminal_reached_at=now；
3) cleanup_due_at=now+retention_days, cleanup_status="pending"；
4) 后台任务分页清理到期记录。

[验收标准]
1) 第1/2次失败能返回个性化英文补充建议；
2) 第3次失败直接终态文案，不再调用LLM；
3) 无邮件发送代码改动；
4) DB migration只新增字段；
5) 两次失败后记录体积明显下降（大JSON字段已置空）；
6) 前端能正确展示轮次和终态。
```

---

## 方案设计

### 1. 整体流程图

```
用户输入文本
    ↓
粗筛失败 (matches.length === 0)
    ↓
调用 AI特征诊断服务
    ↓
LLM分析：识别已有特征 & 推荐缺失特征
    ↓
生成个性化提示文字
    ↓
渲染到前端（match_failed.html）
    ↓
用户根据建议补充信息并重新提交（最多2次）
    ↓
若第2次补充后仍失败 → 返回终态固定文案（不实现邮件发送）
```

### 2. 核心模块设计

#### 2.1 特征类型库（Feature Catalog）

定义系统认可的所有特征类型，分为以下几大类。**注意**："物品属性类"中缺失的特征为最高优先级，应首先提示用户补充。

```
特征类型库结构（按提示优先级）：

【优先级1 - 最高】物品属性类 (CRITICAL)
│   ├── category（物品类别）
│   ├── color（颜色）
│   ├── material（材质）
│   ├── brand（品牌）
│   └── size（尺寸）
│   [缺失这些特征会优先提示用户补充]
│
├── 【优先级2】物品特征类
│   ├── distinguishing_marks（特殊标记/纹理）
│   ├── damage_condition（损坏状况）
│   ├── accessories（配件）
│   └── pattern（图案）
│
├── 【优先级3】地点信息类
│   ├── location（丢失地点）
│   ├── building_name（建筑名称）
│   └── floor_room（楼层/房间）
│
├── 【优先级4】时间信息类
│   ├── approximate_date（大约日期）
│   ├── time_of_day（时段）
│   └── time_context（时间背景）
│
└── 【优先级5】其他信息类
    ├── previous_owner_info（原主人信息）
    └── special_circumstances（特殊情况）
```

#### 2.2 功能模块

**创建新文件** `app/search/diagnostic.py`

包含主要函数：

```python
async def analyze_item_features(
    user_text: str,
    llm_model: str = "glm-4-flash",
    api_key: Optional[str] = None
) -> Dict:
    """
    分析用户输入文本的特征
    
    返回结构：
    {
        "identified_features": {
            "category": "backpack",
            "color": "black",
            ...
        },
        "missing_attributes": ["brand", "material"],  # 物品属性类缺失，最高优先级
        "missing_features": ["distinguishing_marks"],  # 其他特征类缺失
        "suggestions": [
            "Please add the brand or model if you remember it",
            "Any damage or special marks could help us find it",
            ...
        ],
        "raw_llm_output": "..."  # 用于调试
    }
    """
```

#### 2.3 LLM Prompt设计

**特征识别Prompt（多语言输入，英文输出）**：

```
You are an expert Lost-and-Found analyzer.
The user input may be in any language. You must output English only.

Task:
1) Extract explicitly mentioned features.
2) Identify missing attributes from Product Attributes first:
     category, color, material, brand, size.
3) Identify missing features from other categories.
4) Return strict JSON only.

Output JSON schema:
{
    "identified_features": {
        "<feature_type>": "<extracted_value>"
    },
    "missing_attributes": ["category|color|material|brand|size"],
    "missing_features": ["<other_feature_type>"],
    "suggestions": ["English suggestion 1", "English suggestion 2"]
}

Rules:
- Do not infer unsupported facts.
- Prioritize missing_attributes in suggestions.
- Suggestions must be short and actionable.
- Output JSON only, no markdown, no extra text.

User description: "{user_text}"
```

### 3. 集成点

#### 位置1：后端路由 - search_router

**文件**: `app/search/routes.py`

**修改位置**: `/search/screen` 端点

当粗筛失败时（`matches.length === 0`）：

```python
# assist_round: 0=首次失败, 1=第一次补充后失败, 2=第二次补充后失败
assist_round = int(form.get("assist_round", 0))

if not matches:
    if assist_round >= 2:
        # 终态回复：不再继续AI补充
        return JSONResponse({
            "screening": screening,
            "diagnostic": None,
            "terminal": {
                "reached": True,
                "message": "Item is not currently in our database. We will notify you by email if a matching record appears."
            }
        })

    try:
        diagnostic_result = await analyze_item_features(
            user_text=text,
            db=db
        )
        # 将诊断结果返回给前端（包含优先级化的缺失特征建议）
    except Exception as e:
        # Fallback到原有固定提示
        diagnostic_result = None

return JSONResponse({
    "screening": screening,
    "diagnostic": diagnostic_result,  # 新增字段，包含缺失特征诊断
    "terminal": {"reached": False},
    "assist_round_next": assist_round + 1,
    "outcome": {...}
})
```

#### 位置2：前端展示 - match_failed.html

**文件**: `templates/search/match_failed.html`

**修改内容**：

1. 用诊断结果中的建议替换固定提示文字
2. 展示已识别的特征
3. 高亮展示缺失的特征
4. 提供交互式补充界面
5. 增加补充轮次控制（最多2次）并在终态时显示固定英文回复

**示例UI**：
```
已识别的特征:
✓ 物品类型: 黑色背包
✓ 颜色: 黑色

缺失的关键特征 (我们建议补充):
⚠ 品牌/型号
⚠ 材质  
⚠ 特殊标记或损坏
⚠ 丢失时间
⚠ 丢失地点详情

您的补充建议:
- 如果您记得品牌或型号，请告诉我们
- 任何损坏、标签或特殊标记都可能帮助我们找到它
- ...
```

#### 位置3：用户提交处理

**文件**: `app/search/routes.py` - `/match/failed/submit` 端点

当用户最终提交新Request时：

```python
# 存储诊断结果供后续匹配参考
record.diagnostic_data = json.dumps(updated_diagnostic)

# 记录补充轮次，防止无限补充循环
record.assist_round = assist_round
```

#### 位置4：终态固定回复

**文件**: `app/search/routes.py`

当 `assist_round >= 2` 且仍无匹配时：

1. 立即返回固定英文终态文案：
    - `Item is not currently in our database. We will notify you by email if a matching record appears.`
2. 不再继续调用AI补充接口
3. 不实现邮件发送功能（仅返回上述固定文案）
4. 触发数据清理流程（防止数据库膨胀）：
    - 立即清空/置空大字段：`diagnostic_data`、`identified_features`、`missing_features`
    - 写入清理计划时间：`cleanup_due_at = now + retention_days`
    - 由后台清理任务在到期后删除终态无匹配记录（或转归档表）

---

## 技术架构

### 4.1 新增模块依赖

```
app/
├── search/
│   ├── routes.py          (修改：集成诊断调用)
│   ├── match.py           (现有)
│   ├── ranking.py         (现有)
│   └── diagnostic.py      (新增：特征诊断引擎)
│
service/
├── schemas/
│   └── diagnostic.py      (新增：Pydantic模型)
│
templates/
└── search/
    └── match_failed.html  (修改：展示诊断结果)
```

### 4.2 数据表修改（可选，仅允许新增字段，不做删改）

在 `RequestModel` 添加字段存储诊断结果：

```python
class RequestModel(Base):
    # 现有字段...
    diagnostic_data: str = Column(String, nullable=True)  # JSON格式
    identified_features: str = Column(String, nullable=True)  # JSON格式
    missing_features: str = Column(String, nullable=True)  # JSON格式
    assist_round: int = Column(Integer, default=0)  # AI补充轮次，最大2
    terminal_reached: bool = Column(Boolean, default=False)  # 是否已进入终态
    terminal_reached_at: datetime = Column(DateTime, nullable=True)  # 进入终态时间
    cleanup_due_at: datetime = Column(DateTime, nullable=True)  # 到期清理时间
    cleanup_status: str = Column(String(20), default="pending")  # pending/cleaned/archived
```

清理策略（建议）：

1. `assist_round >= 2` 且无匹配时，立即执行轻量瘦身（将大JSON字段置空）。
2. 设置 `cleanup_due_at`（例如 7~30 天）。
3. 后台定时任务按批次清理 `terminal_reached=true AND cleanup_due_at<=now()` 的记录。
4. 若需审计，先归档到历史表，再删除主表记录。
5. 所有清理操作均使用分页/限流，避免锁表和性能抖动。

---

## 实现步骤

### 执行顺序（必须）

1. 先改后端 `app/search/routes.py`：加 `assist_round` 分支与终态返回。
2. 再建 `app/search/diagnostic.py`：实现诊断函数与LLM调用。
3. 再改前端 `templates/search/match_failed.html`：轮次显示与终态展示。
4. 最后做模型字段新增与清理任务。
5. 每一步都要跑最小可用验证后再进行下一步。

### 阶段1：核心特征诊断引擎（Week 1）

- [ ] 确定最终的特征类型库
- [ ] 创建 `app/search/diagnostic.py`
- [ ] 设计和优化LLM Prompt
- [ ] 实现 `analyze_item_features()` 函数
- [ ] 单元测试

### 阶段2：后端集成（Week 1-2）

- [ ] 修改 `app/search/routes.py` 的 `/search/screen` 端点
- [ ] 添加诊断调用逻辑
- [ ] 添加 `assist_round` 计数与上限判断（最多2次）
- [ ] 实现终态固定英文回复返回
- [ ] 实现错误处理和Fallback机制
- [ ] 性能测试（API调用延迟）

### 阶段3：前端适配（Week 2）

- [ ] 修改 `templates/search/match_failed.html`
- [ ] 展示已识别特征
- [ ] 高亮显示缺失特征建议
- [ ] 增加补充轮次提示（第1次/第2次）
- [ ] 第2次后失败时展示终态结论，不再请求AI补充
- [ ] 优化UX和样式

### 阶段4：数据持久化和优化（Week 2-3）

- [ ] 可选：扩展数据库模型存储诊断结果
- [ ] 增加 `assist_round` 字段
- [ ] 增加终态与清理字段（`terminal_reached`、`terminal_reached_at`、`cleanup_due_at`、`cleanup_status`）
- [ ] 两次失败后立即清空大字段，降低单行存储体积
- [ ] 增加后台清理任务（批量清理到期终态记录）
- [ ] 实现缓存机制避免重复诊断
- [ ] 监控和日志记录

### 阶段5：测试和上线（Week 3-4）

- [ ] 端到端集成测试
- [ ] 用户测试反馈
- [ ] A/B测试对比效果
- [ ] 上线和监控

---

## 特征类型库详细定义

**需要用户提供**：

1. **完整的特征类型列表**
   - 各类型的英文名称和中文描述
   - 示例值

2. **物品属性类必填策略**
    - 在 `category/color/material/brand/size` 中，哪些必须优先提示

3. **语言规则确认**
    - 输入支持多语言
    - 输出统一英文

---

## API调用成本考虑

| 场景 | 调用频率 | 成本 | 优化方法 |
|------|---------|------|---------|
| 粗筛失败后诊断 | 低 | 低 | 只在失败时调用 |
| 用户最终提交 | 低 | 低 | 缓存重复查询 |
| 后台批处理 | 可配置 | 可控 | 批量API调用 |

---

## 风险评估与缓解方案

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| API调用超时 | 用户体验 | 设置超时限制，Fallback到固定提示 |
| LLM识别错误 | 错误建议 | 用户可忽略建议，手动输入 |
| 成本超支 | 运营成本 | 添加配额限制和监控告警 |
| 延迟增加 | 用户体验 | 异步处理，使用缓存 |
| 补充轮次失控 | 用户体验与成本 | 服务端强制 `assist_round<=2`，超过直接终态返回 |
| 数据膨胀 | 存储与性能 | 两次失败后字段瘦身 + 到期批量清理 + 可选归档 |

---

## 与现有系统的兼容性

✅ **完全向后兼容**：
- 如果诊断失败，自动Fallback到原有固定提示
- 不影响其他搜索功能
- 数据库变更遵循“只增不减”原则（仅新增字段，不删除既有字段）

---

## 下一步

1. **确认特征类型库**：用户提供详细的特征列表（英文名称、中文描述、示例值）
2. **优化Prompt**：根据特征库调整LLM提示词
3. **多语言测试**：测试各种语言输入的识别准确性
4. **原型测试**：在测试环境验证诊断效果
5. **迭代改进**：基于测试结果优化建议优先级策略

---

**方案评分**: ⭐⭐⭐⭐⭐ (5/5)
- 技术可行性：5/5
- 现有基础：5/5  
- 用户价值：4.5/5
- 实施复杂度：3/5 (中等，可管理)
