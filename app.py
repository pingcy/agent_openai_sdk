# 导入必要的库和模块
from agents import (
    Agent, 
    Runner, 
    AsyncOpenAI, 
    OpenAIChatCompletionsModel,
    ModelSettings,
    ModelTracing,
    RunConfig,trace,
    WebSearchTool,RunContextWrapper
) 
from openai.types.responses import ResponseTextDeltaEvent
from agents import function_tool,input_guardrail,GuardrailFunctionOutput,InputGuardrailTripwireTriggered
from agents.tracing import set_tracing_disabled,set_trace_processors
import asyncio
from typing_extensions import TypedDict, Any
import httpx
from tools_functions import calculator
from tools_llamaindex import search_web,rag_query
from pydantic import BaseModel
from dataclasses import dataclass
import pprint
import os
import json
from pathlib import Path
import logfire

# 配置日志
logfire.configure(console=False)
logfire.instrument_openai_agents()

# 初始化 OpenAI 客户端
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 初始化模型
model_doubao = OpenAIChatCompletionsModel(
    model='doubao-1.5-32k',
    openai_client=openai_client
)

model_openai = OpenAIChatCompletionsModel(
    model='gpt-4o-mini',
    openai_client=openai_client
)

# 使用默认模型
model = model_openai

# 数据模型定义
class Answer(BaseModel):
    '''
    用于定义回答的数据结构
    origin_question: 原始问题
    answer_chinese: 中文回答
    answer_english: 英文回答
    source: 回答参考知识来源
    '''
    origin_question: str
    answer_chinese: str
    answer_english: str
    source: str

@dataclass
class UserInfo:
    UserId: str
    UserName: str

class SensitiveCheckOutput(BaseModel):
    is_sensitive: bool
    reasoning: str

# 内容审核防护栏
input_guardrail_agent = Agent(
    name="内容审核",
    instructions="""检查用户输入是否包含敏感的政治话题。
    如果内容包含以下内容，返回true：
    - 有争议的政治讨论
    - 敏感的地缘政治问题
    - 极端政治观点
    请为决定提供简短的理由。""",
    model=model_openai,
    output_type=SensitiveCheckOutput,
)

@input_guardrail
async def input_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str 
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_sensitive,
    )

# 创建智能体实例
math_agent = Agent(
    name="MathAssistant",
    instructions="你是一个数学助手，专门解答数学相关的问题",
    model=model,
    tools=[calculator],
)

refine_agent = Agent[UserInfo](
    name="RefineAssistant",
    instructions="请根据输入的问题,生成3个细化的相关问题,不需要解释，只需要列出问题",
    model=model,
    tools=[]
)

main_agent = Agent[UserInfo](
    name="MainAssistant",
    instructions="通过中英文双语回答来协助用户。如果询问数学问题,请交给MathAssistant。其他问题可借助工具完成。 ",
    model=model,
    handoffs = [math_agent],
    tools = [refine_agent.as_tool(tool_name="refine_question",tool_description="负责细化问题，以获得更多细节"),
                search_web,
                rag_query],
    input_guardrails=[input_guardrail],
    output_type=Answer
)

rate_agent = Agent(
    name="RateAssistant",
    instructions="你是一个裁判。会根据输入的对话记录，对最终答案进行评价与打分, 评分范围为1-5",
    model=model
)

# 处理用户查询
async def process_query(user_input, user_info):
    """处理用户输入并返回智能体的响应"""

    try:
        # 设置跟踪处理
        set_trace_processors([])
        
        with trace(workflow_name="Test Workflow"):   
            result = await Runner.run(
                main_agent, user_input,
                context=user_info
            )
            
            # 评价结果
            rating_result = await Runner.run(
                rate_agent, 
                result.to_input_list() + [{"role": "user", "content": "请给出你对上述回答的评价"}]
            )
            
            return {
                "main_result": result,
                "rating": rating_result
            }
        
    except InputGuardrailTripwireTriggered as e:
        return {
            "error": "input_blocked",
            "reason": e.guardrail_result.output.output_info.reasoning
        }
    except Exception as e:
        return {
            "error": "general_error",
            "message": str(e)
        }

# 界面显示结果
def display_results(results):
    """格式化并显示智能体的输出结果"""
    if "error" in results:
        if results["error"] == "input_blocked":
            print(f"\n⚠️ 输入被拦截【{results['reason']}】")
        else:
            print(f"\n❌ 发生错误: {results['message']}")
        return
    
    main_result = results["main_result"]
    rating = results["rating"]
    
    print(f'\n🤖 AI(Last Agent: {main_result.last_agent.name}): ')
    pprint.pprint(main_result.final_output)
    
    print(f'\n⭐ 评价(RateAssistant): ')
    pprint.pprint(rating.final_output)

# 显示欢迎信息
def show_welcome():
    """显示欢迎信息和使用说明"""
    print("\n" + "="*50)
    print("🤖 欢迎使用 AI 助手!")
    print("💡 输入您的问题，系统将尝试回答")
    print("❓ 输入 'quit' 退出程序")
    print("="*50 + "\n")

async def main_async():
    """异步主程序入口"""
    show_welcome()
    
    user = UserInfo('ID001', '张三')
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入你的问题 (输入 'quit' 退出): ").strip()
            
            if user_input.lower() == 'quit':
                print("感谢使用，再见! 👋")
                break
            
            if not user_input:
                print("输入不能为空，请重新输入")
                continue
            
            # 处理用户查询
            results = await process_query(user_input, user)
            
            # 显示结果
            display_results(results)
                
        except KeyboardInterrupt:
            print("\n程序已被用户中断")
            break

def main():
    """主程序入口"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()