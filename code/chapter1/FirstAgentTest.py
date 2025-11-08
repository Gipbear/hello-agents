"""
智能旅行助手

使用前请确保已设置以下环境变量：
- LLM_API_KEY: LLM服务的API密钥
- LLM_BASE_URL: LLM服务的基础URL
- LLM_MODEL_ID: 使用的模型名称
- TAVILY_API_KEY: Tavily搜索API的密钥

或者运行 setup_env.sh 脚本自动设置环境变量
"""

import os
import re
import sys
import requests
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from tavily import TavilyClient
from loguru import logger


# 配置loguru日志
logger.remove()  # 移除默认处理器
logger.add(
    sink=sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)


@dataclass
class Config:
    """配置类，管理所有环境变量配置"""
    api_key: str
    base_url: str
    model_id: str
    tavily_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'Config':
        """从环境变量加载配置"""
        return cls(
            api_key=os.environ.get("LLM_API_KEY", "not_needed"),
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
            model_id=os.environ.get("LLM_MODEL_ID", "qwen2.5:7b"),
            tavily_api_key=os.environ.get("TAVILY_API_KEY")
        )

    def validate(self) -> bool:
        """验证配置是否有效"""
        if self.api_key == "not_needed" and not self.base_url.startswith("http"):
            return False
        return True


class WeatherService:
    """天气查询服务"""

    @staticmethod
    def get_weather(city: str) -> str:
        """
        通过调用 wttr.in API 查询真实的天气信息

        Args:
            city: 城市名称

        Returns:
            天气信息字符串
        """
        url = f"https://wttr.in/{city}?format=j1"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 提取当前天气状况
            current_condition = data['current_condition'][0]
            weather_desc = current_condition['weatherDesc'][0]['value']
            temp_c = current_condition['temp_C']

            return f"{city}当前天气：{weather_desc}，气温{temp_c}摄氏度"

        except requests.exceptions.RequestException as e:
            return f"错误：查询天气时遇到网络问题 - {e}"
        except (KeyError, IndexError) as e:
            return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"


class AttractionService:
    """景点推荐服务"""

    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key

    def get_attraction(self, city: str, weather: str) -> str:
        """
        根据城市和天气，搜索并返回景点推荐

        Args:
            city: 城市名称
            weather: 天气状况

        Returns:
            景点推荐字符串
        """
        if not self.tavily_api_key:
            return "错误：未配置TAVILY_API_KEY。请设置环境变量TAVILY_API_KEY。"

        return self._get_tavily_attraction(city, weather)

    def _get_tavily_attraction(self, city: str, weather: str) -> str:
        """使用Tavily API获取景点推荐"""
        try:
            tavily = TavilyClient(api_key=self.tavily_api_key)
            query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

            response = tavily.search(query=query, search_depth="basic", include_answer=True)

            if response.get("answer"):
                return response["answer"]

            # 格式化原始结果
            formatted_results = []
            for result in response.get("results", []):
                formatted_results.append(f"- {result['title']}: {result['content']}")

            if not formatted_results:
                return "抱歉，没有找到相关的旅游景点推荐。"

            return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)

        except Exception as e:
            return f"错误：执行Tavily搜索时出现问题 - {e}"


class LLMClient:
    """大语言模型客户端"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        调用LLM生成回应

        Args:
            prompt: 用户提示
            system_prompt: 系统提示词

        Returns:
            生成的文本
        """
        logger.info("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                stream=False,
                timeout=60
            )
            answer = response.choices[0].message.content
            logger.info("大语言模型响应成功。")
            return answer
        except Exception as e:
            logger.error("调用LLM API时发生错误: {}", e)
            return "错误：调用语言模型服务时出错。"


class ActionParser:
    """动作解析器"""

    @staticmethod
    def parse_action(llm_output: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, str]]]:
        """
        解析LLM输出的动作

        Args:
            llm_output: LLM输出文本

        Returns:
            (action_type, tool_name, kwargs) 元组
        """
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            return None, None, None

        action_str = action_match.group(1).strip()

        # 检查是否是完成动作
        if action_str.startswith("finish"):
            finish_match = re.search(r'finish\(answer="(.*)"\)', action_str)
            if finish_match:
                return "finish", finish_match.group(1), None
            return None, None, None

        # 解析工具调用
        tool_match = re.search(r"(\w+)\((.*)\)", action_str)
        if not tool_match:
            return None, None, None

        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        return "tool", tool_name, kwargs


class TravelAgent:
    """智能旅行助手主类"""

    # 系统提示词
    SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动。
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须在 Action 中使用 `finish(answer="...")` 来输出最终答案。

请开始吧！
"""

    def __init__(self, config: Config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.weather_service = WeatherService()
        self.attraction_service = AttractionService(config.tavily_api_key)

        # 可用工具映射
        self.available_tools = {
            "get_weather": self.weather_service.get_weather,
            "get_attraction": self.attraction_service.get_attraction,
        }

    def run(self, user_prompt: str, max_loops: int = 5) -> str:
        """
        运行智能体

        Args:
            user_prompt: 用户请求
            max_loops: 最大循环次数

        Returns:
            最终答案
        """
        logger.info("用户输入: {}\n{}", user_prompt, "="*40)

        prompt_history = [f"用户请求: {user_prompt}"]

        for i in range(max_loops):
            logger.info("--- 循环 {} ---\n", i+1)

            # 构建完整提示
            full_prompt = "\n".join(prompt_history)

            # 调用LLM进行思考
            llm_output = self.llm_client.generate(full_prompt, self.SYSTEM_PROMPT)
            logger.info("模型输出:\n{}\n", llm_output)
            prompt_history.append(llm_output)

            # 解析并执行动作
            action_type, action_data, kwargs = ActionParser.parse_action(llm_output)

            if action_type is None:
                logger.error("解析错误：模型输出中未找到有效的 Action。")
                break

            if action_type == "finish":
                logger.info("任务完成，最终答案: {}", action_data)
                return action_data

            if action_type == "tool":
                if action_data in self.available_tools:
                    observation = self.available_tools[action_data](**kwargs)
                else:
                    observation = f"错误：未定义的工具 '{action_data}'"

                # 记录观察结果
                observation_str = f"Observation: {observation}"
                logger.info("{}\n{}", observation_str, "="*40)
                prompt_history.append(observation_str)

        logger.error("错误：达到最大循环次数，任务未完成。")
        return "错误：达到最大循环次数，任务未完成。"


def main():
    """主函数"""
    logger.info("=== 智能旅行助手 ===")

    # 加载配置
    config = Config.from_env()

    # 显示配置信息
    logger.info("配置信息:")
    logger.info(f"  模型: {config.model_id}")
    logger.info(f"  服务地址: {config.base_url}")
    logger.info(f"  API密钥: {'已设置' if config.api_key != 'not_needed' else '使用默认值'}")
    logger.info(f"  Tavily API: {'已设置' if config.tavily_api_key else '未设置'}")

    # 验证配置
    if not config.validate():
        logger.error("错误：配置无效，请检查环境变量设置")
        return

    # 创建智能体
    agent = TravelAgent(config)

    # 用户请求
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"

    # 运行智能体
    try:
        result = agent.run(user_prompt)
        logger.info("最终结果: {}", result)
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error("程序运行出错: {}", e)


if __name__ == "__main__":
    main()