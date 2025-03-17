from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,ModelSettings,ModelTracing,RunConfig,WebSearchTool
from agents import function_tool,RunContextWrapper
import asyncio
from typing_extensions import TypedDict, Any
import httpx
from typing import List, Optional
from dataclasses import dataclass

class SearchResult(TypedDict):
    title: str
    url: str
    content: str
    publishedDate: str

@dataclass
class UserInfo:
    UserId: str
    UserName: str

@function_tool
def calculator(wrapper: RunContextWrapper[UserInfo],expression: str) -> float:
    """
    计算简单的四则运算表达式
    
    Args:
        expression: 数学表达式字符串，如 "1 + 2" 或 "3 * 4"
    
    Returns:
        float: 计算结果
    """

    print(f'Start calculation for {wrapper.context.UserName}: {expression}...')

    try:
        # 移除所有空格
        expression = expression.replace(' ', '')
        # 仅允许数字和基本运算符
        if not all(c in '0123456789+-*/.()' for c in expression):
            raise ValueError("Invalid characters in expression")
        return float(eval(expression))
    except Exception as e:
        print(f"Calculation error: {e}")
        return 0.0
