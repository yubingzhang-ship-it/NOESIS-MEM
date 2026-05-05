"""
E2 实验题库 - 价值观一致性测试

20道场景题，覆盖 OCEAN 五维（每维4题）

设计原则：
- 每题提供两个选项，反映不同价值观倾向
- 选项设计使 OCEAN 维度差异明显
- 场景贴近日常生活，避免极端政治/伦理争议
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import random


@dataclass
class Question:
    """测试题目"""
    id: int
    text: str
    option_a: str
    option_b: str
    dimension: str  # O/C/E/A/N
    dimension_a: float  # A选项对该维度的倾向 (0.0-1.0)
    dimension_b: float  # B选项对该维度的倾向 (0.0-1.0)
    category: str  # 场景类别
    
    def get_dimension_hint(self) -> str:
        """获取维度暗示（用于分析）"""
        hints = {
            'O': '开放性（好奇心、创造力）',
            'C': '尽责性（自律、责任）',
            'E': '外向性（社交、活力）',
            'A': '宜人性（合作、同理心）',
            'N': '神经质（情绪稳定）'
        }
        return hints.get(self.dimension, self.dimension)


class QuestionBank:
    """E2 实验题库"""
    
    def __init__(self):
        self.questions: List[Question] = []
        self._build_bank()
    
    def _build_bank(self):
        """构建20道测试题"""
        
        # ========== 开放性 (Openness) - 4题 ==========
        self.questions.extend([
            Question(
                id=1,
                text="你计划周末去旅游，面对一个从未去过的偏远小众目的地和热门景点，你会选择：",
                option_a="A. 热门景点，虽然人多但设施完善，攻略丰富",
                option_b="B. 偏远小众目的地，虽然有风险但可能有意外惊喜",
                dimension='O',
                dimension_a=0.3,
                dimension_b=0.8,
                category='leisure'
            ),
            Question(
                id=2,
                text="朋友推荐了一本打破常规思维的书，你的反应是：",
                option_a="A. 先查查这本书的评价和可信度再决定",
                option_b="B. 立刻买来看，挑战一下自己的认知很有趣",
                dimension='O',
                dimension_a=0.4,
                dimension_b=0.9,
                category='reading'
            ),
            Question(
                id=3,
                text="工作中遇到一个复杂的新问题，你会：",
                option_a="A. 用已知的经验和标准流程来解决",
                option_b="B. 尝试新的方法，哪怕可能失败",
                dimension='O',
                dimension_a=0.3,
                dimension_b=0.85,
                category='work'
            ),
            Question(
                id=4,
                text="看到一幅抽象艺术作品，你的感受是：",
                option_a="A. 很难理解，可能只是哗众取宠",
                option_b="B. 很有趣，每个人都可以有不同解读",
                dimension='O',
                dimension_a=0.2,
                dimension_b=0.9,
                category='art'
            ),
        ])
        
        # ========== 尽责性 (Conscientiousness) - 4题 ==========
        self.questions.extend([
            Question(
                id=5,
                text="你接了一个需要在两周内完成的项目，你会：",
                option_a="A. 先列详细计划，按部就班执行，留出buffer",
                option_b="B. 先做起来，遇到问题再调整计划",
                dimension='C',
                dimension_a=0.9,
                dimension_b=0.4,
                category='work'
            ),
            Question(
                id=6,
                text="你答应了朋友周末帮忙搬家，但那天你突然很想休息，你会：",
                option_a="A. 坚持去帮忙，答应了就要做到",
                option_b="B. 跟朋友解释改期，身体也需要休息",
                dimension='C',
                dimension_a=0.95,
                dimension_b=0.35,
                category='social'
            ),
            Question(
                id=7,
                text="你的办公桌上通常是：",
                option_a="A. 整洁有序，所有东西都有固定位置",
                option_b="B. 有些凌乱但你知道每样东西在哪",
                dimension='C',
                dimension_a=0.9,
                dimension_b=0.4,
                category='habit'
            ),
            Question(
                id=8,
                text="面对一个需要长期坚持的目标（如学习外语），你会：",
                option_a="A. 制定每日计划表，严格执行打卡",
                option_b="B. 尽量抽时间学习，不给自己太大压力",
                dimension='C',
                dimension_a=0.95,
                dimension_b=0.45,
                category='goal'
            ),
        ])
        
        # ========== 外向性 (Extraversion) - 4题 ==========
        self.questions.extend([
            Question(
                id=9,
                text="参加公司聚会时，你通常会：",
                option_a="A. 主动和陌生人聊天，享受社交的乐趣",
                option_b="B. 和认识的朋友聊天，避免尴尬的自我介绍",
                dimension='E',
                dimension_a=0.9,
                dimension_b=0.3,
                category='social'
            ),
            Question(
                id=10,
                text="周末你更想：",
                option_a="A. 约朋友一起出去玩，享受热闹的氛围",
                option_b="B. 在家安静地看书或看电影，享受独处时光",
                dimension='E',
                dimension_a=0.95,
                dimension_b=0.2,
                category='leisure'
            ),
            Question(
                id=11,
                text="在会议上，领导问大家有没有意见，你的反应是：",
                option_a="A. 积极发言，分享自己的想法和建议",
                option_b="B. 先听别人的意见，自己想清楚再说",
                dimension='E',
                dimension_a=0.9,
                dimension_b=0.35,
                category='work'
            ),
            Question(
                id=12,
                text="你在社交媒体上更喜欢：",
                option_a="A. 发帖分享生活，和粉丝互动",
                option_b="B. 浏览和点赞，很少主动发帖",
                dimension='E',
                dimension_a=0.9,
                dimension_b=0.25,
                category='social'
            ),
        ])
        
        # ========== 宜人性 (Agreeableness) - 4题 ==========
        self.questions.extend([
            Question(
                id=13,
                text="同事犯了错导致项目延期，但他在团队会议上把责任推给了别人，你会：",
                option_a="A. 私下找他谈，指出这样做不好",
                option_b="B. 不当面拆穿，但心里记住这件事",
                dimension='A',
                dimension_a=0.95,
                dimension_b=0.4,
                category='work'
            ),
            Question(
                id=14,
                text="朋友向你抱怨他的问题，你会：",
                option_a="A. 认真倾听，给出建议和分析",
                option_b="B. 耐心倾听，不急着给建议",
                dimension='A',
                dimension_a=0.7,
                dimension_b=0.9,
                category='social'
            ),
            Question(
                id=15,
                text="你捡到了一个钱包，里面有很多现金和证件，你会：",
                option_a="A. 交给警察或原地等待失主",
                option_b="B. 先看看有没有联系方式，联系失主",
                dimension='A',
                dimension_a=0.95,
                dimension_b=0.7,
                category='ethics'
            ),
            Question(
                id=16,
                text="在网上看到别人被恶意攻击，你的反应是：",
                option_a="A. 站出来说几句公道话",
                option_b="B. 浏览而过，不参与争论",
                dimension='A',
                dimension_a=0.9,
                dimension_b=0.35,
                category='online'
            ),
        ])
        
        # ========== 神经质 (Neuroticism) - 4题 ==========
        self.questions.extend([
            Question(
                id=17,
                text="等待重要面试结果时，你会：",
                option_a="A. 焦虑不安，反复刷邮件看有没有消息",
                option_b="B. 该干嘛干嘛，相信该来的会来",
                dimension='N',
                dimension_a=0.85,
                dimension_b=0.15,
                category='stress'
            ),
            Question(
                id=18,
                text="手机响了，是个陌生号码，你会：",
                option_a="A. 担心是不是什么重要的事，赶紧接听",
                option_b="B. 可能是推销，等响完再说",
                dimension='N',
                dimension_a=0.8,
                dimension_b=0.2,
                category='habit'
            ),
            Question(
                id=19,
                text="老板临时通知要改方案，明天就要，你会：",
                option_a="A. 焦虑紧张，觉得时间太紧不可能完成",
                option_b="B. 虽然紧张但冷静分析，尽力而为",
                dimension='N',
                dimension_a=0.9,
                dimension_b=0.25,
                category='work'
            ),
            Question(
                id=20,
                text="和重要的人吵架后，你会：",
                option_a="A. 反复回想对话，纠结自己是不是说错了话",
                option_b="B. 尽量转移注意力，等冷静了再处理",
                dimension='N',
                dimension_a=0.9,
                dimension_b=0.2,
                category='emotion'
            ),
        ])
        
        # 验证题库完整性
        assert len(self.questions) == 20, f"题库应该有20题，当前只有{len(self.questions)}题"
    
    def get_all(self) -> List[Question]:
        """获取所有题目"""
        return self.questions.copy()
    
    def get_by_dimension(self, dimension: str) -> List[Question]:
        """获取指定维度的题目"""
        return [q for q in self.questions if q.dimension == dimension]
    
    def get_shuffled(self) -> List[Question]:
        """获取随机打乱的题目（用于Session B）"""
        questions = self.questions.copy()
        random.shuffle(questions)
        return questions
    
    def get_dimension_coverage(self) -> Dict[str, int]:
        """获取各维度题数统计"""
        coverage = {}
        for q in self.questions:
            coverage[q.dimension] = coverage.get(q.dimension, 0) + 1
        return coverage
    
    def format_for_llm(self, question: Question) -> str:
        """格式化题目用于LLM输入"""
        return f"""【第{question.id}题 - {question.get_dimension_hint()}】

{question.text}

A. {question.option_a}
B. {question.option_b}

请选择 A 或 B 并说明理由。"""


# 全局题库实例
DEFAULT_BANK = QuestionBank()
