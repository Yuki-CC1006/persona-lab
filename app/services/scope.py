"""Hard scope gate: a relationship profile, not a general-purpose assistant.

The product promise is 「会拒答的关系画像」. That promise cannot be delegated to
the model: with an API key present the prompt used to say "answer normally even
without memories", so out-of-scope questions (quantum computing, write me some
code) got answered anyway. The golden set expects a refusal, and only the
offline path refused — which made the CI numbers misleading.

So the decision is made here, deterministically and locally, before any model
call. Order matters and is deliberate:

1. greetings and short follow-ups in an existing thread are never refused;
2. any in-scope signal wins over an out-of-scope cue
   ("帮我写一段给她的话" is about her, so it is in scope);
3. an explicit out-of-scope cue is refused, even inside a follow-up
   ("那帮我写一段快速排序的代码" is a new request, not a continuation);
4. nothing recognised is refused, with guidance on how to ask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus
from app.services.retrieval import search_memories_fts

REFUSAL_MESSAGE = (
    "I can't help with that. I only keep track of things between you and the "
    "other person — I'm not a general assistant, so general knowledge or coding "
    "questions are better handled elsewhere. If you want to ask about them, or "
    "talk through how you two interact, I'm here."
)

# Words that only make sense when talking about the person this profile is about.
# Bilingual on purpose: the profile may hold Chinese chat logs, but questions may
# arrive in English.
PORTRAIT_VOCAB = re.compile(
    r"她|他|我们|咱|对方|对象|女朋友|男朋友|另一半|伴侣|爱人|老婆|老公|姑娘|妹子|女神|"
    r"喜欢|讨厌|不爱|爱吃|口味|忌口|吃|喝|喝茶|咖啡|菜|饭|零食|酒|"
    r"习惯|脾气|性格|心情|情绪|开心|高兴|难过|生气|委屈|介意|雷区|底线|害怕|怕|紧张|焦虑|压力|"
    r"爱好|兴趣|日常|作息|睡|起床|熬夜|逛|电影|剧|音乐|歌|书|游戏|运动|健身|旅行|去哪|玩|"
    r"约|见面|约会|聊天|回复|消息|微信|电话|沟通|表达|语气|态度|想法|感受|"
    r"打算|计划|安排|未来|考研|实习|工作|上班|学习|考试|毕业|搬家|"
    r"生日|纪念日|节日|礼物|惊喜|送|表白|求婚|"
    r"关系|相处|在一起|冷战|吵架|和好|分手|异地|信任|安全感|边界|"
    r"家人|父母|妈妈|爸爸|朋友|闺蜜|同事|舍友|老师|"
    r"住|城市|老家|哪里人|健康|生病|感冒|发烧|医院|体检|"
    r"花钱|消费|预算|省钱|存钱|"
    r"\b(?:she|her|hers|he|him|his|they|them|we|us|partner|girlfriend|boyfriend|"
    r"wife|husband|fiance|fiancee|spouse)\b|"
    r"\b(?:like|likes|liked|love|loves|loved|dislike|dislikes|hate|hates|prefer|prefers|"
    r"favorite|favourite|taste|allergy|allergic)\b|"
    r"\b(?:food|drink|drinks|coffee|tea|meal|dinner|lunch|breakfast|snack|snacks|"
    r"alcohol|spicy|sweet)\b|"
    r"\b(?:habit|habits|personality|mood|temper|feeling|feelings|happy|sad|angry|upset|"
    r"anxious|stressed|sensitive|boundary|boundaries|afraid|scared)\b|"
    r"\b(?:hobby|hobbies|interest|interests|routine|schedule|sleep|sleeps|wake|wakes|"
    r"movie|movies|show|shows|music|song|songs|book|books|game|games|sport|sports|"
    r"gym|travel|trip)\b|"
    r"\b(?:date|dating|meet|meeting|chat|chats|reply|replies|message|messages|wechat|"
    r"phone|call|talk|tone|attitude|thoughts)\b|"
    r"\b(?:plan|plans|planning|future|work|job|study|studying|exam|exams|graduate|"
    r"graduation|move|moving)\b|"
    r"\b(?:birthday|anniversary|holiday|gift|gifts|surprise|confess|propose|proposal)\b|"
    r"\b(?:relationship|together|argument|fight|breakup|long distance|trust|security)\b|"
    r"\b(?:family|parents|mom|mum|dad|father|mother|friend|friends|colleague|colleagues|"
    r"roommate|teacher)\b|"
    r"\b(?:live|lives|living|city|hometown|health|sick|ill|fever|hospital|checkup)\b|"
    r"\b(?:money|spend|spending|budget|save|saving)\b",
    re.IGNORECASE,
)

# Domains and task requests that have nothing to do with this profile.
FOREIGN_CUE = re.compile(
    r"帮我写|帮我做|写一段|写个|写一个|生成一段|翻译|续写|改写|润色|"
    r"帮我总结|总结一下这|概括一下这|提取一下这|做个表|写代码|代码|编程|算法|"
    r"排序|查找|时间复杂度|正则表达式|sql|python|java|javascript|c\+\+|html|css|"
    r"编译器|服务器|部署|docker|kubernetes|api|接口文档|报错|异常|堆栈|日志|"
    r"量子|退相干|相对论|微积分|积分|导数|微分|方程|公式|定理|证明题|"
    r"化学|分子|原子|化合物|物理|生物|细胞|基因|天文|黑洞|星系|"
    r"汇率|股票|基金|财报|报税|税率|贷款|利率|"
    r"法律条文|劳动合同|合同条款|病历|诊断|症状|用药|处方|"
    r"\b(?:code|coding|programming|script|algorithm|sorting|quicksort|mergesort|"
    r"regex|regular expression|sql|python|java|javascript|html|css|api|docker|"
    r"kubernetes|compiler|server|deploy|deployment|stack trace|exception|debug)\b|"
    r"\b(?:quantum|decoherence|relativity|calculus|integral|derivative|equation|"
    r"formula|theorem|proof)\b|"
    r"\b(?:chemistry|molecule|atom|compound|physics|biology|cell|gene|genetics|"
    r"astronomy|black hole|galaxy)\b|"
    r"\b(?:exchange rate|stock|stocks|fund|funds|earnings|tax|taxes|loan|loans|"
    r"interest rate|mortgage)\b|"
    r"\b(?:legal|contract|labour law|labor law|medical record|diagnosis|symptom|"
    r"symptoms|prescription|medication)\b|"
    r"\b(?:write me|write a|write an|generate a|generate an|translate|summarize|"
    r"summarise|proofread|rewrite|paraphrase|make a table|create a table)\b",
    re.IGNORECASE,
)

GREETING = re.compile(
    r"^\s*(你好|您好|嗨|哈喽|在吗|在不在|早上好|中午好|晚上好|晚安|"
    r"谢谢|多谢|感谢|好的|好嘞|嗯+|收到|"
    r"hi|hello|hey|ok|okay|thanks|thank you|thx|ty|good morning|good afternoon|"
    r"good evening|good night|yeah|yep|yup|sure|alright|got it|noted|cool|nice)"
    r"[啊呀呢吧吗。！!?？~～,.\s]*$",
    re.IGNORECASE,
)

# Short continuations make sense only inside an existing conversation.
FOLLOWUP_START = re.compile(r"^\s*(那|那呢|这个|那个|哪个|为什么|为啥|还有|继续|然后|所以|"
                            r"是吗|真的|展开|多说|再说|比如|例如|再来|怎么|多久|多少|"
                            r"嗯|好的|对|不是|可以|不行|"
                            r"what about|how about|and|also|then|so|why|what|which|"
                            r"when|where|who|how|more|go on|continue|for example|"
                            r"really|expand|tell me more|right|ok|okay|sure|no|yes|"
                            r"not really)",
                            re.IGNORECASE)

# A short follow-up is not enough to guess a topic from, so cap its length.
# English continuations are longer by nature than their Chinese equivalents.
FOLLOWUP_MAX_CHARS = 40


@dataclass(frozen=True)
class ScopeVerdict:
    in_scope: bool
    reason: str
    message: str = ""

    @property
    def refusal(self) -> str:
        return self.message or REFUSAL_MESSAGE


def known_subjects(db: Session) -> list[str]:
    """Names that appear as subjects of confirmed memories."""
    rows = (
        db.query(Memory.subject)
        .filter(
            Memory.status == MemoryStatus.CONFIRMED.value,
            Memory.subject.isnot(None),
        )
        .distinct()
        .all()
    )
    return [str(name).strip() for (name,) in rows if name and str(name).strip()]


def classify_scope(db: Session, text: str, *, history: list[dict] | None = None) -> ScopeVerdict:
    """Decide locally whether ``text`` is about this profile."""
    query = (text or "").strip()
    if not query:
        # Nothing to judge; leave it to the normal path.
        return ScopeVerdict(True, "empty")

    if GREETING.match(query):
        return ScopeVerdict(True, "greeting")

    if _mentions_subject(db, query):
        return ScopeVerdict(True, "subject")
    if PORTRAIT_VOCAB.search(query):
        return ScopeVerdict(True, "portrait-vocab")
    if _has_lexical_evidence(db, query):
        return ScopeVerdict(True, "knowledge-base")

    if FOREIGN_CUE.search(query):
        return ScopeVerdict(False, "foreign-cue")

    if history and len(query) <= FOLLOWUP_MAX_CHARS and FOLLOWUP_START.match(query):
        # A continuation of an ongoing conversation about the profile.
        return ScopeVerdict(True, "follow-up")

    return ScopeVerdict(False, "unrecognised")


def _mentions_subject(db: Session, query: str) -> bool:
    return any(name and name in query for name in known_subjects(db))


# The FTS query ORs every token, so common English function words would make
# almost any sentence look like a lexical match. The gate only counts content
# words as evidence.
_GATE_STOPWORDS = frozenset(
    "a an the and or but if then than that this these those there here is are was were am "
    "be been being do does did done doing have has had having will would shall should can "
    "could may might must i me my mine we us our ours you your yours he him his she her hers "
    "it its they them their theirs what which who whom whose when where why how not no nor "
    "so to of in on at for with about from by as into over after before again more most some "
    "any all just also very really only own same too now please".split()
)

_CJK_PAT = re.compile(r"[\u4e00-\u9fff]")


def _content_terms(query: str) -> list[str]:
    """Query terms that carry meaning, for the lexical-evidence check."""
    if _CJK_PAT.search(query):
        return [query]
    terms = re.findall(r"[A-Za-z0-9']+", query)
    return [t for t in terms if len(t) > 1 and t.lower() not in _GATE_STOPWORDS]


def _has_lexical_evidence(db: Session, query: str) -> bool:
    """True only for a real lexical overlap.

    Semantic hits are deliberately not used here: the local hash embedding
    returns low-score noise (~0.05) for unrelated questions, which would let
    everything through the gate.
    """
    try:
        terms = _content_terms(query)
        if not terms:
            return False
        return bool(search_memories_fts(db, " ".join(terms), limit=1))
    except Exception:  # noqa: BLE001 — a broken index must not block answers
        return False
