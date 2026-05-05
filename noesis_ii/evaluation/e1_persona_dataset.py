"""
E1 实验评估数据集

包含20条中文对话文本，每条已有人工标注的 OCEAN 五维分数。
用于验证人格提取准确性。
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AnnotatedDialogue:
    """标注对话"""
    id: int
    text: str
    ocean_label: Dict[str, float]  # 人工标注的真实分数
    source: str  # 来源说明


# E1 评估数据集（60条：40条中文 + 20条英文）
E1_DATASET: List[AnnotatedDialogue] = [
    # ========== 高开放性（中文） ==========
    AnnotatedDialogue(
        id=1,
        text="最近我在读一本关于量子物理的科普书，虽然很多概念很难懂，但越看越觉得这个世界太神奇了。我还在考虑要不要去学一门新的编程语言，感觉AI发展太快，不学习就要被淘汰了。",
        ocean_label={'O': 0.85, 'C': 0.50, 'E': 0.45, 'A': 0.60, 'N': 0.40},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=2,
        text="上周去了一趟景德镇，第一次体验了陶瓷制作，那种泥土在手中成型的感觉太治愈了。回来后我开始关注各种手工艺，想学学木工或者皮具制作。",
        ocean_label={'O': 0.90, 'C': 0.40, 'E': 0.55, 'A': 0.70, 'N': 0.25},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=3,
        text="我对星座和塔罗牌一直很感兴趣，最近在学紫微斗数。虽然很多人说是迷信，但我认为传统文化里有很多值得研究的东西，不能一竿子打死。",
        ocean_label={'O': 0.88, 'C': 0.35, 'E': 0.50, 'A': 0.55, 'N': 0.45},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=4,
        text="我最近在研究思维导图和记忆宫殿，觉得这些方法太神奇了。还报名了一个心理学课程，想了解人类认知的奥秘。",
        ocean_label={'O': 0.87, 'C': 0.60, 'E': 0.45, 'A': 0.65, 'N': 0.35},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=5,
        text="我喜欢尝试各种新鲜事物，最近迷上了手冲咖啡，研究不同豆子的风味和冲泡方法。还想去学冲浪，挑战一下自己。",
        ocean_label={'O': 0.92, 'C': 0.45, 'E': 0.60, 'A': 0.60, 'N': 0.30},
        source="自编测试"
    ),
    
    # ========== 高尽责性（中文） ==========
    AnnotatedDialogue(
        id=6,
        text="我每天早上6点起床，先跑步半小时，然后做早餐。七点开始学习编程，晚上十点准时睡觉。周末会提前规划好下周的任务清单，已经坚持了两年多了。",
        ocean_label={'O': 0.45, 'C': 0.95, 'E': 0.30, 'A': 0.55, 'N': 0.35},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=7,
        text="我做任何事情都习惯先列计划，甚至出去玩都要提前做好攻略。工作中我从不拖延，答应了的事情一定会按时完成，宁可加班也不会敷衍。",
        ocean_label={'O': 0.40, 'C': 0.92, 'E': 0.35, 'A': 0.65, 'N': 0.50},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=8,
        text="我的桌面永远整整齐齐，每样东西都有固定的位置。我用 Notion 管理生活的方方面面，从学习计划到购物清单，全都记录得清清楚楚。",
        ocean_label={'O': 0.50, 'C': 0.90, 'E': 0.25, 'A': 0.50, 'N': 0.30},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=9,
        text="我每个月都会做预算，严格控制支出。工作项目我都会制定详细的时间表，每完成一个任务就打勾，确保不会遗漏任何细节。",
        ocean_label={'O': 0.35, 'C': 0.94, 'E': 0.30, 'A': 0.55, 'N': 0.40},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=10,
        text="我坚持每天阅读1小时，不管多忙都会抽出时间。每年都会设定明确的目标，分为短期、中期和长期，定期复盘进度。",
        ocean_label={'O': 0.55, 'C': 0.91, 'E': 0.35, 'A': 0.60, 'N': 0.35},
        source="自编测试"
    ),
    
    # ========== 高外向性（中文） ==========
    AnnotatedDialogue(
        id=11,
        text="周末刚参加了一个陌生人聚会，认识了好多有趣的朋友！我是那种见到新面孔就想搭话的人，在公司也是茶水间的话题担当。打算下个月组织一场徒步活动。",
        ocean_label={'O': 0.60, 'C': 0.55, 'E': 0.95, 'A': 0.75, 'N': 0.30},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=12,
        text="我超级喜欢和人聊天，不管是出租车司机还是理发师，我都能聊半天。朋友圈里我发得最勤，直播我也经常看，感觉一个人待着太无聊了。",
        ocean_label={'O': 0.55, 'C': 0.40, 'E': 0.92, 'A': 0.70, 'N': 0.40},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=13,
        text="开会的时候我总是第一个发言，有新想法就忍不住要分享。公司团建我肯定是积极分子，K歌之王说的就是我。下周有个行业论坛，我报了名要去分享经验。",
        ocean_label={'O': 0.65, 'C': 0.60, 'E': 0.90, 'A': 0.60, 'N': 0.35},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=14,
        text="我喜欢组织各种活动，最近刚办了一场同学聚会，来了30多个人。平时没事就喜欢约朋友出来吃饭聊天，觉得和人相处特别开心。",
        ocean_label={'O': 0.60, 'C': 0.55, 'E': 0.93, 'A': 0.70, 'N': 0.30},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=15,
        text="我在社交媒体上很活跃，经常发视频和帖子，粉丝有好几万。线下也是聚会的核心人物，大家都说我性格开朗，很有感染力。",
        ocean_label={'O': 0.65, 'C': 0.45, 'E': 0.94, 'A': 0.65, 'N': 0.35},
        source="自编测试"
    ),
    
    # ========== 高宜人性（中文） ==========
    AnnotatedDialogue(
        id=16,
        text="我很少和人起冲突，遇到分歧一般会先退一步。我觉得大家都不容易，何必为难别人呢？借钱给朋友我从来不好意思催，感觉伤感情。",
        ocean_label={'O': 0.45, 'C': 0.50, 'E': 0.55, 'A': 0.92, 'N': 0.35},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=17,
        text="我是那种看到流浪猫狗就忍不住要喂的人。工作上从来不争名夺利，宁可自己吃亏也不想让别人难堪。朋友找我倾诉，我能听一整晚不带烦的。",
        ocean_label={'O': 0.50, 'C': 0.45, 'E': 0.50, 'A': 0.95, 'N': 0.25},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=18,
        text="网上看到有人被喷，我总会忍不住站出来说几句公道话。现实中遇到插队的我也敢当面制止。虽然可能被说多管闲事，但我觉得该说就得说。",
        ocean_label={'O': 0.55, 'C': 0.60, 'E': 0.45, 'A': 0.88, 'N': 0.40},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=19,
        text="我总是先考虑别人的感受，朋友有困难我会第一时间帮忙。工作中我经常主动承担额外的任务，觉得团队合作最重要。",
        ocean_label={'O': 0.50, 'C': 0.65, 'E': 0.55, 'A': 0.93, 'N': 0.30},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=20,
        text="我喜欢帮助别人，经常参加志愿者活动。看到别人开心我就觉得很满足，从来不会计较付出多少。",
        ocean_label={'O': 0.45, 'C': 0.55, 'E': 0.60, 'A': 0.94, 'N': 0.25},
        source="自编测试"
    ),
    
    # ========== 高神经质（中文） ==========
    AnnotatedDialogue(
        id=21,
        text="我这个人特别容易焦虑，收到领导邮件就担心是不是做错了什么。晚上睡前总是忍不住刷手机看有没有新消息，半夜经常醒来想事情。出个远门要检查好几遍行李。",
        ocean_label={'O': 0.45, 'C': 0.55, 'E': 0.35, 'A': 0.50, 'N': 0.90},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=22,
        text="我特别敏感，别人一个眼神不对我就想是不是我哪里做错了。和朋友吵架后能纠结好几天，反复回想自己是不是说错话了。容易低落，也容易胡思乱想。",
        ocean_label={'O': 0.50, 'C': 0.45, 'E': 0.30, 'A': 0.55, 'N': 0.88},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=23,
        text="我承认自己有点完美主义，事情不做到最好就心里不舒服。压力大的时候会失眠，有时候会因为一点小事就绪崩溃大哭。但我正在学习调节自己。",
        ocean_label={'O': 0.55, 'C': 0.75, 'E': 0.30, 'A': 0.45, 'N': 0.85},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=24,
        text="我总是担心不好的事情发生，比如家人健康、工作失误等。遇到重要事情前会紧张得吃不下饭，手心出汗。",
        ocean_label={'O': 0.40, 'C': 0.60, 'E': 0.35, 'A': 0.55, 'N': 0.89},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=25,
        text="我对批评特别敏感，哪怕是善意的建议也会让我难过很久。经常会想别人是不是在背后说我坏话，容易陷入自我怀疑。",
        ocean_label={'O': 0.45, 'C': 0.50, 'E': 0.30, 'A': 0.60, 'N': 0.91},
        source="自编测试"
    ),
    
    # ========== 低 OCEAN (中性/混合)（中文） ==========
    AnnotatedDialogue(
        id=26,
        text="我这人比较随性，想做什么就做什么，不太喜欢被规则束缚。房间乱点无所谓，只要自己知道东西在哪就行。工作上完成任务就行，不想太卷。",
        ocean_label={'O': 0.35, 'C': 0.30, 'E': 0.45, 'A': 0.50, 'N': 0.45},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=27,
        text="我比较务实，只相信看得见摸得着的东西。什么星座命理我觉得都是胡扯，做决定主要靠逻辑分析。讨厌空谈，喜欢实实在在的结果。",
        ocean_label={'O': 0.25, 'C': 0.60, 'E': 0.35, 'A': 0.45, 'N': 0.40},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=28,
        text="我比较慢热，不会主动和人搭话，更享受一个人安静地待着。社交对我来说是消耗能量的事，每次聚会完都想赶紧回家休息。",
        ocean_label={'O': 0.45, 'C': 0.50, 'E': 0.20, 'A': 0.55, 'N': 0.50},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=29,
        text="我比较直接，有话就直说，不太会拐弯抹角。遇到看不惯的事我会当面指出，不会背后说人。有些人觉得我说话太冲，但我就是这样的性格。",
        ocean_label={'O': 0.50, 'C': 0.55, 'E': 0.40, 'A': 0.30, 'N': 0.45},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=30,
        text="我这人心态挺好的，不容易焦虑也不容易激动。天塌下来有高个子顶着，着急也没用。遇到问题就解决问题，解决不了就接受，不跟自己过不去。",
        ocean_label={'O': 0.50, 'C': 0.45, 'E': 0.50, 'A': 0.55, 'N': 0.20},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=31,
        text="我一般不会主动尝试新鲜事物，但也不排斥。生活过得平平淡淡，没有特别大的起伏。对人对事都保持中立态度，不偏激。",
        ocean_label={'O': 0.50, 'C': 0.50, 'E': 0.45, 'A': 0.50, 'N': 0.45},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=32,
        text="我有时候会拖延，但重要的事情还是会按时完成。社交上不算活跃，但也有几个要好的朋友。情绪比较稳定，很少大喜大悲。",
        ocean_label={'O': 0.45, 'C': 0.60, 'E': 0.40, 'A': 0.55, 'N': 0.40},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=33,
        text="我对新观念持开放态度，但也会保持理性判断。工作上认真负责，但不会过度追求完美。和人相处友好，但也会保持适当距离。",
        ocean_label={'O': 0.60, 'C': 0.65, 'E': 0.50, 'A': 0.60, 'N': 0.35},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=34,
        text="我喜欢有计划地生活，但偶尔也会即兴做些事情。社交中会主动交流，但更享受高质量的独处时间。情绪偶尔会波动，但很快就能调整过来。",
        ocean_label={'O': 0.55, 'C': 0.60, 'E': 0.55, 'A': 0.55, 'N': 0.45},
        source="自编测试"
    ),
    AnnotatedDialogue(
        id=35,
        text="我对自己要求不高，过得去就行。和人相处比较随和，不会刻意讨好。遇到挫折会有些沮丧，但很快就能恢复。",
        ocean_label={'O': 0.40, 'C': 0.45, 'E': 0.45, 'A': 0.60, 'N': 0.40},
        source="自编测试"
    ),
    
    # ========== 英文标注集（20条） ==========
    # 高开放性
    AnnotatedDialogue(
        id=36,
        text="I've been reading a lot about astrophysics lately, and it's mind-blowing how vast the universe is. I'm also thinking of learning how to play the guitar, even though I've never had any musical training before.",
        ocean_label={'O': 0.88, 'C': 0.50, 'E': 0.45, 'A': 0.60, 'N': 0.35},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=37,
        text="Last weekend I went to a modern art exhibition and was completely fascinated by the abstract paintings. I'm considering taking an art class to explore my creative side.",
        ocean_label={'O': 0.90, 'C': 0.45, 'E': 0.55, 'A': 0.70, 'N': 0.25},
        source="英文自编测试"
    ),
    
    # 高尽责性
    AnnotatedDialogue(
        id=38,
        text="I wake up at 5:30 every morning, meditate for 15 minutes, then go for a run. I plan my entire week on Sunday evenings and stick to my schedule rigorously.",
        ocean_label={'O': 0.40, 'C': 0.95, 'E': 0.30, 'A': 0.55, 'N': 0.30},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=39,
        text="I keep a detailed journal of all my tasks and accomplishments. I never miss deadlines and always go the extra mile to ensure my work is perfect.",
        ocean_label={'O': 0.45, 'C': 0.92, 'E': 0.35, 'A': 0.60, 'N': 0.45},
        source="英文自编测试"
    ),
    
    # 高外向性
    AnnotatedDialogue(
        id=40,
        text="I love meeting new people and making friends. I'm always the life of the party and enjoy organizing social events for my friends.",
        ocean_label={'O': 0.60, 'C': 0.55, 'E': 0.95, 'A': 0.75, 'N': 0.30},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=41,
        text="I'm very outgoing and love talking to strangers. I'm active on social media and enjoy being the center of attention.",
        ocean_label={'O': 0.55, 'C': 0.45, 'E': 0.93, 'A': 0.70, 'N': 0.35},
        source="英文自编测试"
    ),
    
    # 高宜人性
    AnnotatedDialogue(
        id=42,
        text="I always try to see things from other people's perspectives and avoid conflicts. I'm happy to help others even if it means sacrificing my own time.",
        ocean_label={'O': 0.50, 'C': 0.55, 'E': 0.55, 'A': 0.92, 'N': 0.30},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=43,
        text="I'm very compassionate and often volunteer at the local animal shelter. I believe in treating everyone with kindness and respect.",
        ocean_label={'O': 0.55, 'C': 0.60, 'E': 0.50, 'A': 0.95, 'N': 0.25},
        source="英文自编测试"
    ),
    
    # 高神经质
    AnnotatedDialogue(
        id=44,
        text="I worry a lot about things that might go wrong. I often lie awake at night thinking about work or personal issues.",
        ocean_label={'O': 0.45, 'C': 0.50, 'E': 0.35, 'A': 0.55, 'N': 0.90},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=45,
        text="I'm very sensitive to criticism and often take things personally. I get anxious easily and have trouble relaxing.",
        ocean_label={'O': 0.50, 'C': 0.45, 'E': 0.30, 'A': 0.60, 'N': 0.88},
        source="英文自编测试"
    ),
    
    # 低 OCEAN (中性/混合)
    AnnotatedDialogue(
        id=46,
        text="I'm pretty laid-back and go with the flow. I don't stress too much about things and prefer to keep my life simple.",
        ocean_label={'O': 0.45, 'C': 0.40, 'E': 0.45, 'A': 0.55, 'N': 0.35},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=47,
        text="I'm practical and focus on what's real. I make decisions based on logic rather than emotions and prefer concrete results over abstract ideas.",
        ocean_label={'O': 0.30, 'C': 0.65, 'E': 0.40, 'A': 0.50, 'N': 0.35},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=48,
        text="I'm introverted and prefer spending time alone or with a small group of close friends. I find social situations draining.",
        ocean_label={'O': 0.45, 'C': 0.55, 'E': 0.25, 'A': 0.60, 'N': 0.45},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=49,
        text="I'm straightforward and say what I think. I don't beat around the bush and prefer honesty even if it might upset people.",
        ocean_label={'O': 0.50, 'C': 0.55, 'E': 0.45, 'A': 0.35, 'N': 0.40},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=50,
        text="I have a balanced approach to life. I'm not too extreme in any personality dimension and try to maintain a moderate perspective.",
        ocean_label={'O': 0.50, 'C': 0.50, 'E': 0.50, 'A': 0.50, 'N': 0.30},
        source="英文自编测试"
    ),
    
    # 混合特质
    AnnotatedDialogue(
        id=51,
        text="I'm creative and enjoy trying new things, but I also value structure and planning in my life.",
        ocean_label={'O': 0.75, 'C': 0.70, 'E': 0.55, 'A': 0.60, 'N': 0.40},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=52,
        text="I'm outgoing and love socializing, but I also need my alone time to recharge.",
        ocean_label={'O': 0.65, 'C': 0.50, 'E': 0.75, 'A': 0.65, 'N': 0.35},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=53,
        text="I'm very organized and detail-oriented, but I can also be spontaneous and enjoy unexpected adventures.",
        ocean_label={'O': 0.60, 'C': 0.80, 'E': 0.55, 'A': 0.55, 'N': 0.30},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=54,
        text="I'm kind and compassionate, but I also know when to stand up for myself and set boundaries.",
        ocean_label={'O': 0.55, 'C': 0.65, 'E': 0.50, 'A': 0.80, 'N': 0.45},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=55,
        text="I'm generally calm and composed, but I can get anxious when faced with significant challenges or changes.",
        ocean_label={'O': 0.50, 'C': 0.60, 'E': 0.45, 'A': 0.65, 'N': 0.60},
        source="英文自编测试"
    ),
    
    # 多样化场景
    AnnotatedDialogue(
        id=56,
        text="I love traveling and experiencing different cultures. I'm always curious about how people live in other parts of the world.",
        ocean_label={'O': 0.85, 'C': 0.55, 'E': 0.65, 'A': 0.70, 'N': 0.35},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=57,
        text="I'm very dedicated to my work and always strive for excellence. I believe in putting in the effort to achieve my goals.",
        ocean_label={'O': 0.45, 'C': 0.85, 'E': 0.40, 'A': 0.55, 'N': 0.40},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=58,
        text="I enjoy meeting new people and making connections. I find networking both fun and valuable for personal growth.",
        ocean_label={'O': 0.65, 'C': 0.55, 'E': 0.85, 'A': 0.70, 'N': 0.30},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=59,
        text="I'm very empathetic and try to help others whenever I can. I believe in the importance of kindness and compassion.",
        ocean_label={'O': 0.55, 'C': 0.50, 'E': 0.55, 'A': 0.90, 'N': 0.30},
        source="英文自编测试"
    ),
    AnnotatedDialogue(
        id=60,
        text="I tend to overthink things and worry about the future. I'm working on being more present and less anxious.",
        ocean_label={'O': 0.50, 'C': 0.55, 'E': 0.35, 'A': 0.55, 'N': 0.85},
        source="英文自编测试"
    ),
]


def get_e1_dataset() -> List[AnnotatedDialogue]:
    """获取 E1 评估数据集"""
    return E1_DATASET.copy()


def get_dimension_distribution() -> Dict[str, Dict[str, int]]:
    """获取维度分布统计"""
    distribution = {dim: {'high': 0, 'medium': 0, 'low': 0} for dim in 'OCEAN'}
    
    for item in E1_DATASET:
        for dim, score in item.ocean_label.items():
            if score >= 0.7:
                distribution[dim]['high'] += 1
            elif score <= 0.4:
                distribution[dim]['low'] += 1
            else:
                distribution[dim]['medium'] += 1
    
    return distribution


# 显示数据集统计
if __name__ == "__main__":
    print("E1 评估数据集统计")
    print("=" * 40)
    print(f"总样本数: {len(E1_DATASET)}")
    print()
    
    dist = get_dimension_distribution()
    dim_names = {'O': '开放性', 'C': '尽责性', 'E': '外向性', 'A': '宜人性', 'N': '神经质'}
    
    for dim in 'OCEAN':
        print(f"{dim} ({dim_names[dim]}):")
        print(f"  高分(≥0.7): {dist[dim]['high']}")
        print(f"  中分: {dist[dim]['medium']}")
        print(f"  低分(≤0.4): {dist[dim]['low']}")
        print()
